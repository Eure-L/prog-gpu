#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h> // gettimeofday
#include <math.h> // pour le calcul d'exponentielle lorsque l'on génère le probleme aléatoire

struct circle {
	// Couleur et opacité
	float r, g, b, alpha;
	// Rayon du disque
	float radius;
	// Centre du disque
	float x, y;
};

// Couleur d'un pixel
// Remarque : on manipule ici des nombres float pour éviter
// les imprécisions liées aux opérations arithmétiques successives sur des
// nombres entiers
struct pixel {
	float r, g, b;
};

// Routine pour générer un fichier au format PPM
// Affichage de la sortie avec la commande "display output.ppm"
static void writeImage(struct pixel *image, int width, int height, const char *filename) {
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
	    	
	        unsigned char r = (unsigned char)(255*image[index].r);
	        unsigned char g = (unsigned char)(255*image[index].g);
	        unsigned char b = (unsigned char)(255*image[index].b);
	
	        fputc(r, f);
	        fputc(g, f);
	        fputc(b, f);
	}            
	fclose(f);
	printf("Wrote image file %s\n", filename);
}

#define DEFAULT_NCIRCLES 3
struct circle default_circles[DEFAULT_NCIRCLES] = {
	{.r = 1.0, .g = 0.0, .b = 0.0, .alpha = 0.8f, .radius=0.2f, .x=0.25, .y=0.5},
	{.r = 0.0, .g = 1.0, .b = 0.0, .alpha = 0.8f, .radius=0.2f, .x=0.5, .y=0.5},
	{.r = 0.0, .g = 0.0, .b = 1.0, .alpha = 0.8f, .radius=0.2f, .x=0.75, .y=0.5}
};


int main(int argc, char **argv)
{
	const float xmin = 0.0f;
	const float xmax = 1.0f;

	const float ymin = 0.0f;
	const float ymax = 1.0f;

	const int width = 800;
	const int height = 800;

	float dx = (xmax - xmin)/(width - 1);
	float dy = (ymax - ymin)/(height - 1);

	int ncircles;
	struct circle *circles;

	if (argc > 1)
	{
		// On initialise une graine pour que l'on obtienne toujours la même séquence aléatoire
		srand(2018);

		// Probleme aleatoire de taille ncircles
		ncircles = atoi(argv[1]);

		// Generation d'un tableau de disques avec des caractéristiques aléatoires
		circles = malloc(ncircles * sizeof(struct circle));
		int i;
		for (i = 0; i < ncircles; i++)
		{
			circles[i].x      = xmin + (xmax - xmin)*drand48();
			circles[i].y      = ymin + (ymax - ymin)*drand48();
			circles[i].radius = 20.0*dx*expf(drand48());;
			circles[i].r      = drand48();
			circles[i].g      = drand48();
			circles[i].b      = drand48();
			circles[i].alpha  = drand48();
		}
	}
	else {
		circles = default_circles;
		ncircles = DEFAULT_NCIRCLES;
	}

	// Il faut bien dessiner quelque part
	struct pixel *image = malloc(width*height*sizeof(struct pixel));
	assert(image);

	int x, y;

	// On initialise le fond avec une couleur blanche
	for (y = 0; y < height; y++)
	for (x = 0; x < width; x++)
	{
		int index = x + y * width;
		image[index].r = 1.0f;
		image[index].g = 1.0f;
		image[index].b = 1.0f;
	}

    	struct timeval start, finish;
        unsigned long long tstart, tfinish;
        gettimeofday(&start, NULL);

	int c;

	// On itere sur les disques
	for (c = 0; c < ncircles; c++)
	{
		// On itere sur les pixels
		for (y = 0; y < height; y++)
		for (x = 0; x < width;  x++)
		{
			float xc = circles[c].x;
			float yc = circles[c].y;
			float rc = circles[c].radius;
			float rc2 = rc*rc;

			// Coordonnées du pixel
			float xpos = xmin + x * dx;
			float ypos = ymin + y * dy;
			float dist2 = (((xpos-xc)*(xpos-xc) + (ypos-yc)*(ypos-yc)));

			// Le pixel est dans le cercle si la distance entre le
			// centre et le pixel est inférieure au rayon
			if (dist2 <= rc2)
			{
				// Caractéristiques du disque
				float alpha = circles[c].alpha;
				float r = circles[c].r;
				float g = circles[c].g;
				float b = circles[c].b;

				int index = x + y * width;

				// Operation de fusion des couleurs
				image[index].r = alpha * r + (1.0f - alpha) * image[index].r;
				image[index].g = alpha * g + (1.0f - alpha) * image[index].g;
				image[index].b = alpha * b + (1.0f - alpha) * image[index].b;
			}
		}
	}

        gettimeofday(&finish, NULL);
        tstart = start.tv_sec * 1000000 + start.tv_usec;
        tfinish = finish.tv_sec * 1000000 + finish.tv_usec;
        printf("Timing : %f ms\n", (tfinish-tstart)*0.001);

    	writeImage(image, width, height, "output.ppm");

	free(image);

	if (argc > 1)
	{
		free(circles);
	}

	return 0;
}
