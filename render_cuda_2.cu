#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h> // gettimeofday
#include <math.h> // pour le calcul d'exponentielle lorsque l'on génère le probleme aléatoire
#include "cuda_safe_call.h"

//ne pas dépasser 1024 en width
#define WIDTH 800
#define HEIGHT 800

#define imgSTRIDE (WIDTH*HEIGHT)

#define cR 0
#define cG 1
#define cB 2
#define cAlpha 3
#define cRadius 4
#define cX 5
#define cY 6

// Routine pour générer un fichier au format PPM
// Affichage de la sortie avec la commande "display output.ppm"
static void writeImage(float *image, int width, int height, const char *filename) {
	FILE *f = fopen(filename, "wb");
	if (!f) {
		perror(filename);
		exit(1);
	}
	
	fprintf(f, "P6\n%d %d\n255\n", width, height);
	for (int y = 0; y < height; ++y)
	for (int x = 0; x < width; ++x)
	{
	    	int index = y * width + x;

			unsigned char r = (unsigned char)(255*image[index]);
	        unsigned char g = (unsigned char)(255*image[index+imgSTRIDE]);
	        unsigned char b = (unsigned char)(255*image[index+imgSTRIDE*2]);
	
	        fputc(r, f);
	        fputc(g, f);
	        fputc(b, f);
	}            
	fclose(f);
	//printf("Wrote image file %s\n", filename);
}

#define DEFAULT_NCIRCLES 3
// struct circle default_circles[DEFAULT_NCIRCLES] = {
// 	{.r = 1.0, .g = 0.0, .b = 0.0, .alpha = 0.8f, .radius=0.2f, .x=0.25, .y=0.5},
// 	{.r = 0.0, .g = 1.0, .b = 0.0, .alpha = 0.8f, .radius=0.2f, .x=0.5, .y=0.5},
// 	{.r = 0.0, .g = 0.0, .b = 1.0, .alpha = 0.8f, .radius=0.2f, .x=0.75, .y=0.5}
// };
// 					
					//		R					G			B				
float default_circles[] = {1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0, 
											 0.8f, 0.8f, 0.8f,  0.2f, 0.2f, 0.2f, 
											0.25, 0.5, 0.75,
											0.5, 0.5, 0.5};


__global__ void computeImage(float * image, float *  circles, const int width, const int height, int ncircles,
								float xmin,float xmax,float ymin, float ymax, float dx,float dy, 
								int xfloor, int xceil, int yfloor, int yceil,
								int circle,float xc, float yc, float rc){
	//int index = threadIdx.x + threadIdx.y * blockDim.x;
	int index = (blockIdx.x + yfloor) * width + (threadIdx.x + xfloor);
	

	float rc2 = rc*rc;

	// printf("%f, %f, %f , %f \n",xc,yc,rc,rc2);

	// Coordonnées du pixel
	int x = threadIdx.x + xfloor;
	int y = blockIdx.x + yfloor;
	float xpos = xmin + x * dx;
	float ypos = ymin + y * dy;
	float dist2 = (((xpos-xc)*(xpos-xc) + (ypos-yc)*(ypos-yc)));

	// Le pixel est dans le cercle si la distance entre le
	// centre et le pixel est inférieure au rayon
	
	if (dist2 <= rc2)
	{
		// Caractéristiques du disque
		float alpha = circles[circle+cAlpha*ncircles];
		float r 	= circles[circle+cR*ncircles];
		float g 	= circles[circle+cG*ncircles];
		float b 	= circles[circle+cB*ncircles];

		// Operation de fusion des couleurs
		image[index] 				= alpha * r + (1.0f - alpha) * image[index];
		image[index+imgSTRIDE]	 	= alpha * g + (1.0f - alpha) * image[index+imgSTRIDE];
		image[index+imgSTRIDE*2] 	= alpha * b + (1.0f - alpha) * image[index+imgSTRIDE*2];
	}
	
}

int main(int argc, char **argv)
{
	const float xmin = 0.0f;
	const float xmax = 1.0f;

	const float ymin = 0.0f;
	const float ymax = 1.0f;

	const int width = WIDTH;
	const int height = HEIGHT;

	float dx = (xmax - xmin)/(width - 1);
	float dy = (ymax - ymin)/(height - 1);

	int ncircles;
	float * circles;

	if (argc > 1)
	{
		// On initialise une graine pour que l'on obtienne toujours la même séquence aléatoire
		srand(2018);

		// Probleme aleatoire de taille ncircles
		ncircles = atoi(argv[1]);

		// Generation d'un tableau de disques avec des caractéristiques aléatoires
		circles = (float *) malloc(ncircles * 7 * sizeof(float));
		int i;
		for (i = 0; i < ncircles; i++)
		{
			circles[i+ncircles*0]	= drand48();
			circles[i+ncircles*1]	= drand48();
			circles[i+ncircles*2]	= drand48();
			circles[i+ncircles*3]	= drand48();
			circles[i+ncircles*4]	= 20.0*dx*expf(drand48());
			circles[i+ncircles*5]	= xmin + (xmax - xmin)*drand48();
			circles[i+ncircles*6]	= ymin + (ymax - ymin)*drand48();
		}
	}
	else {
		circles = default_circles;
		ncircles = DEFAULT_NCIRCLES;
	}
	// Il faut bien dessiner quelque part
	float *image = (float *) malloc(width*height*sizeof(float) * 3);
	assert(image);

	int x, y;

	// On initialise le fond avec une couleur blanche
	for (y = 0; y < height; y++)
	for (x = 0; x < width; x++)
	{
		int index = x + y * width;
		image[index] = 1.0f;
		image[index+imgSTRIDE] = 1.0f;
		image[index+imgSTRIDE*2] = 1.0f;
	}
	struct timeval start, finish;
	unsigned long long tstart, tfinish;
	gettimeofday(&start, NULL);

	// Allocating Device memory
	float * d_image;
	float * d_circles;
	cudaMalloc((float **)&d_image,3*width*height*sizeof(float));
	cudaMalloc((float **)&d_circles,7*ncircles 	* sizeof(float));
	CUDA_SAFE_CALL(cudaMemcpy(d_image, image, 3 * width * height * sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_circles, circles, 7 * ncircles * sizeof(float),cudaMemcpyHostToDevice));
	
	int c;

	// On itere sur les disques
	for (c = 0; c < ncircles; c++)
	{
		float xc = circles[c + cX * ncircles];
		float yc = circles[c + cY * ncircles];
		float rc = circles[c + cRadius * ncircles];
		// Au lieu de cherche si TOUS les pixels de l'image ont une intersection avec le cercle
		// On va chercher seulement si les pixels du carre contenant le cercle ont une intersection avec
		// Pixels y englobant le cercles
		int yfloor  	= (yc - rc < 0) ? 0 : (yc - rc) * height;
		int yceil 		= (yc + rc > 1) ? height : (yc + rc) * height;
		// Pixels X englobant le xercle
		int xfloor  	= (xc - rc < 0) ? 0 : (xc - rc) * width;
		int xceil 		= (xc + rc > 1) ? width : (xc + rc) * width;
 
		
		computeImage<<<yceil-yfloor,xceil-xfloor>>>
			(d_image,d_circles,width,height,ncircles,
			xmin,xmax,ymin,ymax,dx,dy,xfloor,xceil,yfloor,yceil,
			c,xc,yc,rc);
		cudaDeviceSynchronize();
	}
	

	CUDA_SAFE_CALL(cudaDeviceSynchronize()); //Verifier que le code se soit bien lancé sur GPU
	CUDA_SAFE_CALL(cudaMemcpy(image,d_image,3*width*height*sizeof(float),cudaMemcpyDeviceToHost));

	cudaFree(&d_image);
	cudaFree(&d_circles);

	gettimeofday(&finish, NULL);
	tstart = start.tv_sec * 1000000 + start.tv_usec;
	tfinish = finish.tv_sec * 1000000 + finish.tv_usec;
	printf("%f\n", (tfinish-tstart)*0.001);

	writeImage(image, width, height, "output.ppm");

	free(image);

	if (argc > 1)
	{
		free(circles);
	}

	return 0;
}
