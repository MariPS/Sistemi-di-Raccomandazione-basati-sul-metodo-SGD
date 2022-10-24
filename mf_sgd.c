/*
	Seminario:	Sistemi di raccomandazione basati sul metodo SGD
	Autore:		Marika Pia Salvato (mat. 0522500897)
	
	Fattorizzazione di matrice (seriale)
	
	INPUT - nu=0 or nv=0 -> carica il dataset movielens*
		altrimenti   -> genera una matrice sparsa casuale
	
	* Il dataset utilizzato è la versione "ml-latest-small". La cartella del dataset "ml-latest-small" deve trovarsi nella stessa directory di questo file .c
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define DEBUG 0

void generateR(float *R, int nu, int nv, int *n_miss, float *sparsity);
int myRand();
void loadDataset(float *R, int nu, int nv, int *n_miss, float *sparsity);
void train(float *R_train, int n_R_train, float *Rcap, int nu, int nv, int d, float l_r, float reg_param, int n_iter);
float calcPreds(float *U, float *V, int nu, int nv, int d, int utente, int item);
float calcRMSD(float *R_train, int n_R_train, float *Rcap, int nv);

int main(int argc, char const *argv[])
{
    float *R, *R_train;
    int nu, nv, n_miss, d;
    int n_iter, n_R_train;
    int utente, item, r_ui;
    int i,j,c;
    float sparsity, learning_rate = 0.001, reg_param, *Rcap, r_ui_pred, rmsd;
    struct timespec tp_start;
    struct timespec tp_end;

    printf("***\t Fattorizzazione di Matrice - SGD\t***\n\n");
    printf("Dimensioni della matrice di rating R\n\tnu = ");
    scanf("%d", &nu);
    printf("\tnv = ");
    scanf("%d", &nv);


    /*
        Inizializzazione di R
        I ratings vanno da 1 a 5
        Valori uguali a 0 sono i rating mancanti
    */
    if ((nu == 0)||(nv == 0))
    {
        nu=610;
        nv=193609;
        R=(float *)malloc( nu * nv *(sizeof(float)));
        loadDataset(R,nu,nv,&n_miss,&sparsity);

        printf("\nestratto 10x10 di R =\n");
        for ( i = 0; i < 10; i++)
        {
            for ( j = 0; j < 10; j++)
            {
                printf("%.2f\t",R[i*10+j]);
            }
            printf("\n");
        }
        

    }else{
    
        R = (float *)malloc(nu*nv*sizeof(float));
        generateR(R,nu,nv,&n_miss,&sparsity);
        if (nu <= 10 && nv <= 10)
        {
            printf("\nR =\n");
            for ( i = 0; i < nu; i++)
            {
                for ( j = 0; j < nv; j++)
                {
                    printf("%.2f\t", R[i*nv + j]);
                }
                printf("\n");
            }
            printf("\n");   
        }
    
    }
    
    
    printf("\nR dim = (%d,%d),\tsparsità = %.2f%%\n", nu, nv, sparsity);


    /*
        Inizializzazione di Rcap
    */
    Rcap = (float *)malloc(nu*nv*sizeof(float));


    /*
        Creazione del training set a partire da R
        Si selezionano solo i valori diversi da 0.
        R_train è una matrice bidimensionale nx3.
        n = numero totale dei ratings - numero di ratings mancanti 
        Le colonne rappresentano:
        id utente | id item | rating
    */
    
    n_R_train = ((nu*nv)-n_miss);
    R_train = (float *)malloc(n_R_train*3*sizeof(float));
    c = 0;

    for ( i = 0; i < nu; i++)   // i == id utente
    {
        for ( j = 0; j < nv; j++)   // j == id item
        {
            if (R[i*nv+j] != 0)
            {
                R_train[c*3+0] = i;
                R_train[c*3+1] = j;
                R_train[c*3+2] = R[i*nv+j];
                c++; // riga successiva
            }
        }
    }

    if(DEBUG)
    {
        printf("\nR_train (primi 10 elementi)= \nutente\titem\trating\n");
        for ( i = 0; i < 10; i++)
        {
            printf("%.0f\t%.0f\t%.2f\n", R_train[i*3],R_train[i*3+1],R_train[i*3+2]);
        }
    }
    

    /*
        Parametri di training
    */

    printf("\nNumero di fattori latenti d = ");
    scanf("%d", &d);

    printf("Learning rate = ");
    scanf("%f", &learning_rate);

    printf("Parametro di regolarizzazione = ");
    scanf("%f", &reg_param);

    printf("Numero di iterazioni = ");
    scanf("%d", &n_iter);

    // start clock
    clock_gettime(CLOCK_MONOTONIC, &tp_start);

    train(R_train, n_R_train, Rcap, nu, nv, d, learning_rate, reg_param, n_iter);

    // end clock
    clock_gettime(CLOCK_MONOTONIC, &tp_end);
    double time_spent =  (double) (tp_end.tv_sec - tp_start.tv_sec) + 1.0e-9 * (double) (tp_end.tv_nsec - tp_start.tv_nsec);

    if (nu <= 10 && nv <= 10)
    {
        printf("\nR =\n");
        for ( i = 0; i < nu; i++)
        {
            for ( j = 0; j < nv; j++)
            {
                printf("%.2f\t", R[i*nv + j]);
            }
            printf("\n");
        }

        printf("\nRcap =\n");
        for ( i = 0; i < nu; i++)
        {
            for ( j = 0; j < nv; j++)
            {
                printf("%.2f\t", Rcap[i*nv+j]);
            }
            printf("\n");
        }
    }

    /*
        Calcolo di RMSD
    */

    rmsd = calcRMSD(R_train, n_R_train, Rcap, nv);
    printf("\nRMSD = %f\n", rmsd);


    free(R);
    free(Rcap);


    printf("\nTempo di esecuzione = %lf secondi\n", time_spent);

    return 0;
}




void train(float *R_train, int n_R_train, float *Rcap, int nu, int nv, int d, float l_r, float reg_param, int n_iter)
{

    float *U, *V, e=0, r_ui, r_ui_pred=0;
    int r, utente, item;
    int i,j;

    /*
        Inizializzazione delle matrici U e V
        dei fattori latenti degli utenti e degli items 
    */
    U = (float *)malloc(nu*d*sizeof(float));
    V = (float *)malloc(nv*d*sizeof(float));

    srand((int)time(0));
    for ( i = 0; i < nu; i++)
    {
        for ( j = 0; j < d; j++)
        {
            U[i*d + j] = rand() % 2;   // genera numeri da 0 a 1
        }
        
    }

    for ( i = 0; i < d; i++)
    {
        for ( j = 0; j < nv; j++)
        {
            V[i*nv + j] = rand() % 2;   // genera numeri da 0 a 1
        }
    }

    if (DEBUG)
    {
        printf("\nU =\n");
        for ( i = 0; i < nu; i++)
        {
            for ( j = 0; j < d; j++)
            {
                printf("%f\t", U[i*d + j]);
            }
            printf("\n");
        }  
    
        printf("\nV =\n");
        for ( i = 0; i < d; i++)
        {
            for ( j = 0; j < nv; j++)
            {
                printf("%f\t", V[i*nv + j]);
            }
            printf("\n");
        }
        printf("\n");   
    }


    /*
        Inizia la fase di training
    */
    for (int iter = 0; iter < n_iter; iter++)
    {
        // scelta random di 1 rating dal training set
        r = rand() % n_R_train;   // genera numeri da 0 a # dei ratings
        
        utente = (int) R_train[r*3+0]; // id utente
        item = (int) R_train[r*3+1]; // id item
        r_ui = R_train[r*3+2]; // rating

        // calcolo della predizione del rating
        r_ui_pred = calcPreds(U,V,nu,nv,d,utente,item);
        
        // calcolo dell'errore
        e = r_ui - r_ui_pred;
        
        if (n_iter <= 25)
        {
            printf("\niter n.%d\tr(%d,%d)=%.2f,\t r_pred(%d,%d)=%.2f,\t e=%f\n",iter+1, utente,item,r_ui, utente,item,r_ui_pred, e);
        }else if ((iter+1) % 100 == 0)
        {        
            printf("\niter n.%d\tr(%d,%d)=%.2f,\t r_pred(%d,%d)=%.2f,\t e=%f\n",iter+1, utente,item,r_ui, utente,item,r_ui_pred, e);
        }

        // update dei fattori latenti
        for ( i = 0; i < d; i++)
        {
            U[utente*d+i] += l_r * (e*V[i*nv+item] - reg_param*U[utente*d+i]);
            V[i*nv+item] += l_r * (e*U[utente*d+i] - reg_param*V[i*nv+item]);
        }       
    }

    /*
        Generazione di Rcap a partire da U e V
    */
    for ( i = 0; i < nu; i++)
    {
        for ( j = 0; j < nv; j++)
        {

            Rcap[i*nv+j] = calcPreds(U,V,nu,nv,d,i,j);
        } 
    }
    
    free(U);
    free(V);

}

float calcPreds(float *U, float *V, int nu, int nv, int d, int utente, int item)
{
    int i;

    // calcolo della predizione del rating
    float r_ui_pred = 0;
    for ( i = 0; i < d; i++)
    {
        r_ui_pred += U[utente*d+i]*V[i*nv+item];   
    }

    return r_ui_pred;
}

void generateR(float *R, int nu, int nv, int *n_miss, float *sparsity)
{
    int i,j;
    *n_miss=0;
    *sparsity=0;

    srand((int)time(0));

    for ( i = 0; i < nu; i++)
    {
        for ( j = 0; j < nv; j++)
        {
            R[i*nv + j] = myRand();   // genera numeri da 0 a 5
            if(R[i*nv+j] == 0) (*n_miss)++;    // conta i ratings mancanti
        }
        
    }
    
    *sparsity = ((float)(*n_miss)/(float)(nu*nv))*100;

}

int myRand()
{
    /*
        restituisce 0 con prob=75%
        restituisce un numero tra 1 e 5 con prob=25%
    */

    int prob;
    
    prob = rand() % 100+1;
    if (prob <= 75) // se il numero è compreso tra 1 e 75
    {
        return 0;   // resituisce 0
    }
        
    return rand() % 5+1;    // altrimenti resituisce un numero tra 1 e 5
}

float calcRMSD(float *R_train, int n_R_train, float *Rcap, int nv)
{
    int i, utente, item;
    float r_ui_pred, r_ui, rmsd;

    for ( i = 0; i < n_R_train; i++)
    {
        utente = (int) R_train[i*3];
        item = (int) R_train[i*3+1];
        r_ui = R_train[i*3+2];

        r_ui_pred = Rcap[utente*nv+item];

        rmsd += pow(r_ui_pred-r_ui,2);
    }
    rmsd=sqrt(rmsd/n_R_train);

    return rmsd;
}


void loadDataset(float *R, int nu, int nv, int *n_miss, float *sparsity)
{
    *n_miss=0;
    *sparsity=0;

    FILE *fp;
    
    fp = fopen("./ml-latest-small/ratings.csv","r");
    if (fp == NULL)
    {
        printf("\nERRORE nella'apertura del file\n");
        exit(-1);
    } 

    // salta l'intestazione
    fscanf(fp,"%*s");

    int u=0,m=0;
    float r=0;
    
    int i,j, c=0;
    while (fscanf(fp,"%d,%d,%f%*s",&u,&m,&r)==3)
    {
        i=u-1;
        j=m-1;
        R[i*nv+j]=r;
        c++;    // conta quanti ratings sono stati inseriti
    }

    *n_miss=(nu*nv)-c;    // num totale dei ratings - quelli inseriti

    *sparsity = ((float)(*n_miss)/(float)(nu*nv))*100;

    fclose(fp);
}
