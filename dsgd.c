/*
	Seminario:	Sistemi di raccomandazione basati sul metodo SGD
	Autore:		Marika Pia Salvato (mat. 0522500897)
	
	Fattorizzazione di matrice (algoritmo DSGD)
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>
#include<unistd.h>
#include<math.h>

#define DEBUG 0

void generateR(int *R, int nu, int nv, int *n_miss, float *sparsity);
int myRand();
float calcPreds(float *U, float *V, int d, int utente, int item);
float calcRMSD(int *R_train, int n_R_train, float *Rcap, int nv);

int main(int argc, char *argv[])
{
    int *R, nu, nv, *R_train, n_R_train, d, *localR, local_nu, local_nv, local_n_miss=0;
    float *Rcap, *U, *V, *localU, *localV, *traspostaV, *recvV, learning_rate, reg_param;
    int r,utente,item,r_ui;
    float r_ui_pred,e;
    int i,j, j_start, j_stop, n_iter, iter, pattern;

    float T_inizio,T_fine,T_max;
    int me, nproc;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);


    if (me == 0)
    {
        printf("***\t Fattorizzazione di Matrice - DSGD\t***\n\n");
        printf("Dimensioni della matrice di rating R\n\tnu (multiplo di nproc) = ");
        scanf("%d", &nu);
        printf("\tnv (multiiplo di nproc) = ");
        scanf("%d", &nv);


        R = (int *)malloc(nu*nv*sizeof(int));

        /*
            Inizializzazione di R
            I ratings vanno da 1 a 5
            Valori uguali a 0 sono i rating mancanti
        */
        int n_miss=0;
        float sparsity=0;
        generateR(R,nu,nv,&n_miss,&sparsity);

        if (nu <= 10 && nv <= 10)
        {
            printf("\nR =\n");
            for ( i = 0; i < nu; i++)
            {
                for ( j = 0; j < nv; j++)
                {
                    printf("%d\t", R[i*nv + j]);
                }
                printf("\n");
            }
            printf("\n");   
        }
        
        printf("\nR dim = (%d,%d),\tsparsità = %.2f%%\n", nu, nv, sparsity);


        /*
            Inizializzazione di Rcap
        */
        
        Rcap = (float *)malloc(nu*nv*sizeof(float));


        /*
            Parametri di training
        */

        printf("\nNumero di fattori latenti d = ");
        scanf("%d", &d);

        printf("Learning rate = ");
        scanf("%lf", &learning_rate);

        printf("Parametro di regolarizzazione = ");
        scanf("%lf", &reg_param);

        printf("Numero di iterazioni = ");
        scanf("%d", &n_iter);



        /*
            Creazione delle matrici dei fattori latenti U e V
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
                    printf("%lf\t", U[i*d + j]);
                }
                printf("\n");
            }
            printf("\n");   
        }

        if (DEBUG)
        {
            printf("\nV =\n");
            for ( i = 0; i < d; i++)
            {
                for ( j = 0; j < nv; j++)
                {
                    printf("%lf\t", V[i*nv + j]);
                }
                printf("\n");
            }
            printf("\n");   
        }

    }

    // Spedizione dei parametri da parte di 0 a tutti i processori
    MPI_Bcast(&nv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nu, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&learning_rate, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reg_param, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    


    /*
        Distribuzione della matrice R e U
        per blocchi di righe
        Distribuzione di V a blocchi di colonne
    */
    
    // es. se nproc=4 e nu=8, distribuisco 2 righe di R a ogni processo
    // significa assegnare a un processo solo quelle righe di R_train
    // in cui l'id utente corrisponde all'indice di riga di R assegnatogli
    
    local_nu = nu/nproc;
    local_nv = nv/nproc;

    localR = (int *)malloc(local_nu * nv * sizeof(int));

    // P0 invia a tutti la matrice R a blocchi di righe
    MPI_Scatter(
        &R[0], local_nu*nv, MPI_INTEGER,
        &localR[0], local_nu*nv, MPI_INTEGER,
        0, MPI_COMM_WORLD);

    localU = (float *)malloc(local_nu * d * sizeof(float));

    // P0 invia a tutti la matrice U a blocchi di righe
    MPI_Scatter(
        &U[0], local_nu*d, MPI_FLOAT,
        &localU[0], local_nu*d, MPI_FLOAT,
        0, MPI_COMM_WORLD);
    

    // Per poter inviare V a blocchi di colonne
    // P0 deve calcolare la trasposta
    if (me==0)
    {
        traspostaV = (float *)malloc(nv*d*sizeof(float));

        for ( i = 0; i < d; i++)
        {
            for ( j = 0; j < nv; j++)
            {
                traspostaV[j*d+i]=V[i*nv+j];
            }
            
        }
        
    }
    
    // tutti allocano lo spazio per localV
    localV = (float *)malloc(local_nv * d * sizeof(float));
    
    // P0 distribuisce V trasposta per righe
    MPI_Scatter(
        &traspostaV[0], local_nv*d, MPI_FLOAT,
        &localV[0], local_nv*d, MPI_FLOAT,
        0, MPI_COMM_WORLD);


    // in più ogni processo prepara una matrice recvV
    // per il trasferimento del localV durante la SGD
    recvV = (float *)malloc(local_nv*d*sizeof(float));
    // a questo punto, ogni processo ha una porzione di
    //   localU (local_nu x d)
    //   localV (local_nv x d)
    //   localR (local_nu x nv)



    /*
        DSGD
    */
    MPI_Barrier(MPI_COMM_WORLD);
	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio
    
   
    for ( iter = 0; iter < n_iter; iter++)
    {   

        if (me==0)
        {
            printf("\niterazione = %d", iter);
        }


        // per ogni pattern di blocchi indipendenti
        for ( pattern = 0; pattern < nproc; pattern++)
        {
            // ogni processo lavora sempre 
            // sulle stesse righe di R_train e localU
            // cambiano solo le colonne di R_train, localU e V
            // quindi pattern influenza solo le colonne su cui ogni processo deve lavorare
            // es. nproc = 4, nv = 12, col_per_proc = nv/nproc= 3
            // es.  alla iter 1 (pattern 0)
            //          P0 prende le colonne da j=0=(pattern+me)*local_nv a 2(<j+local_nv)   
            //          P1 prende le colonne da j=3=(0+1)*3 a 5 (<3+3)
            //          P2 prende le colonne da j=6=(0+2)*3 a 8 (<6+3)
            //          P3 prende le colonne da j=9=(0+3)*3 a 11 (<9+3)
            //      alla iter 2 (pattern 1)
            //          P0 prende le colonne da j=3=(1+0)*3 a 5 (<3+3)
            //          P1 prende le colonne da j=6=(1+1)*3 a 8 (<6+3)
            //          P2 prende le colonne da j=9=(1+2)*3 a 11 (<9+3)
            //          P3 prende le colonne da j=0=(1+3)*3=12mod12 a 2 (<0+3)
            // in generale, il proc Pi prende le colonne 
            //  da j=((pattern+me)*local_nv)mod(nv) a j+local_nv-1      

            local_n_miss=0;
            j_start = ((pattern+me)*local_nv)%nv;
            j_stop = j_start + local_nv;
            for ( i = 0; i < local_nu; i++)
            {
                for ( j = j_start; j < j_stop; j++)
                {
                    if(localR[i*nv+j]==0) local_n_miss++;
                }
                
            }


            if (local_n_miss < local_nu*local_nv)   //se c'è almeno un rating nel blocco 
            {
                n_R_train = ((local_nu*local_nv)-local_n_miss);
                R_train = (int *)malloc(n_R_train*3*sizeof(int));

                int c = 0;

                for ( i = 0; i < local_nu; i++)   // i == id utente
                {
                    for ( j = j_start; j < j_stop; j++)   // j == id item
                    {
                        if (localR[i*nv+j] != 0)
                        {
                            R_train[c*3+0] = i;
                            // per trasporre l'indice j relativo alla matrice localR (che ha nv colonne)
                            // all'indice relativo al blocco con local_nv colonne
                            R_train[c*3+1] = j-j_start; 
                            R_train[c*3+2] = localR[i*nv+j];
                            c++; // riga successiva
                        }
                    }
                }

              
                // scelta random di 1 rating dal training set
                r = rand() % n_R_train;   // genera numeri da 0 a # dei ratings
            
                utente = R_train[r*3+0]; // id utente
                item = R_train[r*3+1]; // id item è relativo alla colonna in R_
                r_ui = R_train[r*3+2]; // rating

                // calcolo della predizione del rating
                r_ui_pred = calcPreds(localU,localV,d,utente,item);
        
                // calcolo dell'errore
                e = r_ui - r_ui_pred;
                
                // update dei fattori latenti
                if (DEBUG)
                {
                    printf("\nPRIMA (proc %d) pattern=%d \tlocalU =\n",me,pattern);
                    for ( i = 0; i < local_nu; i++)
                    {
                        for ( j = 0; j < d; j++)
                        {
                            printf("%lf\t",localU[i*d+j]);
                        }
                        printf("\n");
                    }
                    
                    printf("\nPRIMA (proc %d) pattern=%d \tlocalV =\n",me,pattern);
                    for ( i = 0; i < local_nv; i++)
                    {
                        for ( j = 0; j < d; j++)
                        {
                            printf("%lf\t",localV[i*d+j]);
                        }
                        printf("\n");
                    }
                }
                
                for ( i = 0; i < d; i++)
                {
                    localU[utente*d+i] += learning_rate * (e*localV[item*d+i] - reg_param*localU[utente*d+i]);
                    localV[item*d+i] += learning_rate * (e*localU[utente*d+i] - reg_param*localV[item*d+i]);
                } 
                if (DEBUG)
                {
                    printf("\nDOPO (proc %d) pattern=%d \tlocalU =\n",me,pattern);
                    for ( i = 0; i < local_nu; i++)
                    {
                        for ( j = 0; j < d; j++)
                        {
                            printf("%lf\t",localU[i*d+j]);
                        }
                        printf("\n");
                    }
                    
                    printf("\nDOPO (proc %d pattern=%d )\tlocalV =\n",me,pattern);
                    for ( i = 0; i < local_nv; i++)
                    {
                        for ( j = 0; j < d; j++)
                        {
                            printf("%lf\t",localV[i*d+j]);
                        }
                        printf("\n");
                    }
                }
            }
            
            // invio delle porzioni di V 
            // ogni processo invia una parte della matrice V appena aggiornata
            // al processo immediatamente precedente nell'anello
            // es. P0 ha finito di processare il blocco0
            //      quindi invia il blocco di V su cui ha lavorato 
            //          (da riga 0 a d, da colonna j_start a j_stop)
            //      al processo P3, che nella iterazione successiva deve lavorare
            //      su quel blocco


            int send = (me==0) ? nproc-1 : (me-1);
            int recv=(me+1)%nproc;

            MPI_Barrier(MPI_COMM_WORLD);
            
            // ogni processo invia il proprio blocco localV al processo precedente nella topologia ad anello
            MPI_Sendrecv(
                &localV[0],local_nv*d,MPI_FLOAT,send,0,
                &recvV[0],local_nv*d,MPI_FLOAT,recv,0,
                MPI_COMM_WORLD,&status);
                
            

            // una volta completata la ricezione, ogni processo sovrascrive il proprio localV con quello appena ricevuto
            for ( i = 0; i < local_nv*d; i++)
            {
                localV[i]=recvV[i];
            }            
            
        }
    }
       

    MPI_Gather(&localU[0],local_nu*d,MPI_FLOAT,&U[0],local_nu*d,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Gather(&localV[0],local_nv*d,MPI_FLOAT,&V[0],local_nv*d,MPI_FLOAT,0,MPI_COMM_WORLD);
            
    // ferma il cronometro
	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine

    MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	MPI_Reduce(&T_fine,&T_max,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

    if (me==0)
    {
        for ( i = 0; i < nu; i++)
        {
            for ( j = 0; j < nv; j++)
            {
                Rcap[i*nv+j] = calcPreds(U,V,d,i,j);
            } 
        }

        if (nu<=12)
        {    
            printf("\n\nRcap = \n");

            for ( i = 0; i < nu; i++)
            {
                for ( j = 0; j < nv; j++)
                {
                printf("%.2f\t",Rcap[i*nv+j]);
                } 
                printf("\n");
            }
        
            printf("\nR =\n");
            for ( i = 0; i < nu; i++)
            {
                for ( j = 0; j < nv; j++)
                {
                printf("%d\t",R[i*nv+j]);
                } 
                printf("\n");
            }    
        }
        
        
        float rmsd = calcRMSD(R_train, n_R_train, Rcap, nv);
        printf("\nRMSD = %f\n", rmsd);

        printf("Processori impegnati: %d\n", nproc);
        printf("Tempo calcolo locale: %lf\n", T_fine);
		printf("MPI_Reduce max time:  %f\n",T_max);

        free(R);
        free(U);
        free(V);
        free(traspostaV);
        free(Rcap);
   }
   

    free(localR);
    free(localU);
    free(localV);
    free(R_train);

    MPI_Finalize();

    return 0;
}

void generateR(int *R, int nu, int nv, int *n_miss, float *sparsity)
{
    int i,j;

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


float calcPreds(float *U, float *V, int d, int utente, int item)
{
    int i;

    // calcolo della predizione del rating
    float r_ui_pred = 0;
    for ( i = 0; i < d; i++)
    {
        r_ui_pred += U[utente*d+i]*V[item*d+i];   
    }

    return r_ui_pred;
}

float calcRMSD(int *R_train, int n_R_train, float *Rcap, int nv)
{
    int i, utente, item,r_ui;
    float r_ui_pred, rmsd=0;

    for ( i = 0; i < n_R_train; i++)
    {
        utente = R_train[i*3];
        item = R_train[i*3+1];
        r_ui = R_train[i*3+2];

        r_ui_pred = Rcap[utente*nv+item];

        rmsd += pow(r_ui_pred-r_ui,2);
    }
    rmsd=sqrt(rmsd/n_R_train);

    return rmsd;
}
