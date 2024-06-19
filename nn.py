import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy
import time

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

#solution exactes
data = scipy.io.loadmat('/home/cycymonkey/Documents/ei2/Projet8/myPINN/data/burgers_shock.mat')
u_vrai = data['usol']

#constantes
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = .01/pi

class PINN(tf.keras.Model):
    """ 
    Définie l'architecture du PINN
    """

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=8, 
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb  
        self.ub = ub
        
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim, 
                                         activation=tf.keras.activations.get(activation))


    @tf.function
    def call(self, X, training = False):
        '''
        Avance forward dans le NN
        '''
        Y = self.scale(X)
        for i in range(self.num_hidden_layers):
            Y = self.hidden[i](Y, training = True)
        output = self.out(Y)
        return(output)
    


class PINNSolver():
    '''
    Définie un solveur en se basant sur un modele déjà défini
    '''

    def __init__(self, model, X_r, gamma = 1, batch_size = 32):
        self.model = model
        self.batch_size = batch_size
        
        self.t = X_r[:,0:1]
        self.x = X_r[:,1:2]

        # historique de loss et nmbre d'iterations
        self.hist_loss = []
        self.epoch = 0
        self.time_epoch = 0
        self.batch = 0
        self.time_batch = 0
        self.gamma = gamma


    def f_r(self, t, x, u, u_t, u_x, u_xx):
        '''
        Partie résiduelle de l'EDP
        '''
        return u_t + u * u_x - viscosity * u_xx
        

    def get_r(self):
        '''
        Calcule le residu de l'EDP
        '''
        batch_t, batch_x = self.iterator.get_next()

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(batch_t)
            tape.watch(batch_x)                
            
            u = self.model(tf.stack([batch_t[:,0], batch_x[:,0]], axis=1), training = True)
                
            u_x = tape.gradient(u, batch_x)
            u_t = tape.gradient(u, batch_t)
            u_xx = tape.gradient(u_x, batch_x)

        del tape

        return self.f_r(batch_t, batch_x, u, u_t, u_x, u_xx)
    

    def loss_f(self, X, u):
        '''
        Calcule de loss
        '''
        r = self.get_r()
        MSEf = tf.reduce_mean(tf.square(r)) 
                
        u_pred = self.model(X, training = True)
        MSEu = tf.reduce_mean(tf.square(tf.subtract(u_pred, u)))
            
        loss = self.gamma*MSEf + MSEu

        return loss
    

    @tf.function
    def get_grad(self, X, u):
        '''
        Calcule le gradient par rapport aux poids et aux biais
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_f(X, u)
            
            grad = tape.gradient(loss, self.model.trainable_variables)
                
        return loss, grad
    

    def create_batches(self, N):
        combined_tensor = tf.concat([self.t, self.x], axis=-1)
        shuffled_combined_tensor = tf.random.shuffle(combined_tensor)

        t_shuf = shuffled_combined_tensor[:, :self.t.shape[-1]]
        x_shuf = shuffled_combined_tensor[:, self.t.shape[-1]:]
    
        batches = tf.data.Dataset.from_tensor_slices((t_shuf, x_shuf)).batch(self.batch_size).repeat(N)
        self.iterator = iter(batches)
        return(int(len(batches)/N))

    
    
    def solve(self, optimizer, X, u, N=1001):
        '''
        optimisation par SGD
        '''
        @tf.function
        def train_step():
            '''
            Réalise un pas de la SGD
            '''
            loss, grad = self.get_grad(X, u)
            
            optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            return loss
        
        nbr_batches = self.create_batches(N)  
        for i in range(N * nbr_batches):

            start_time = time.time()
            loss = train_step()
            end_time = time.time()
            self.time_batch = end_time - start_time

            self.current_loss = loss.numpy()
            self.callback_batch()

            if (i+1) % nbr_batches == 0:
                self.callback_epoch()

    
    def callback_batch(self):
        '''
        Print nombre de batch dans l'epoch, la valeur de loss et temps de calcul
        '''
        self.time_epoch += self.time_batch
        print('It_epoch {:05d}, time_epoch = {:10.4e} s \
              It_batch {:05d}, time_batch = {:10.4e} s, \
              loss = {:10.4e}'.format(self.epoch, self.time_epoch,
                                      self.batch, self.time_batch, 
                                      self.current_loss))
        self.batch += 1



    def callback_epoch(self):
        '''
        Print le nombre d'epoch, la valeur de loss et temps de calcul depuis derniere epoch
        '''
        print('It_epoch {:05d}, time_epoch =  {:10.4e} ,loss = {:10.8e}'.format(self.epoch, self.time_epoch, self.current_loss))
        self.hist_loss.append(self.current_loss)
        self.epoch+=1
        self.time_epoch = 0
        self.batch = 0
        

    def data_prep(self, N_x = 1000, N_t = 1000):
        '''
        Preparation des données pour les plots et la metrique 
        '''
        tspace = np.linspace(self.model.lb[0], self.model.ub[0], N_t+1)
        xspace = np.linspace(self.model.lb[1], self.model.ub[1], N_x+1)

        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(),X.flatten()]).T 
        upred = self.model(tf.cast(Xgrid,DTYPE))
        U = upred.numpy().reshape(N_x+1,N_t+1)
        return(tspace,xspace,U)
    

    def metrique(self):
        '''
        Calcule la racine de l'erreur quadratique réduite entre la prédiction du modèle et la solution exacte 
        '''
        tspace, xspace, U = self.data_prep(N_x=255, N_t=99)
        metrique = np.sqrt(np.mean((U-u_vrai)**2))
        return(metrique)


    def plot_solution(self, nom_save = 'plot_model'):
        '''
        Plot le graphique 
        '''
        tspace, xspace, U = self.data_prep()

        #creation fig et ax + tracé
        fig, ax = plt.subplots()
        fig.set_size_inches(19,12)
        title_style = dict(pad=10, fontname="Ubuntu", fontsize=25)
        img = ax.pcolormesh(tspace, xspace, U, shading="auto", cmap = 'plasma')

        #ajout d'une barre d'echelle
        bar = plt.colorbar(img, ax=ax)

        #ajout label et titre
        ax.set_xlabel('t', fontsize = 20)
        ax.set_ylabel('x', fontsize = 20)
        ax.set_title('u', **title_style)
        plt.savefig("plot/"+nom_save)


    def plot_25_50_75(self, nom_save = 'plot'):
        '''
        Plot 3 graphiques comparant solutions exactes et solution prédites
        '''
        tspace, xspace, U = self.data_prep(N_t = 99, N_x = 255)

        u_t_25=U[:,25]
        u_t_50=U[:,50]
        u_t_75=U[:,25]

        fig, axs = plt.subplots(1,3, figsize=(20,15), sharey=True)
        title_style2 = dict(pad=10, fontname="Ubuntu", fontsize=18)
        axs[0].set_ylabel('u(x,t)', fontsize = 17)

        l1,=axs[0].plot(xspace, u_vrai[:,25], linewidth=6, color='b')
        l2,=axs[0].plot(xspace,u_t_25,linewidth=6,linestyle='dashed',color='r')
        axs[0].set_title('t=0.25', **title_style2)
        axs[0].set_xlabel('x', fontsize=15)

        axs[1].plot(xspace, u_vrai[:,50], linewidth=6, color='b')
        axs[1].plot(xspace, u_t_50, linewidth=6, linestyle='dashed', color='r')
        axs[1].set_title('t=0.50', **title_style2)
        axs[1].set_xlabel('x', fontsize = 15)

        axs[2].plot(xspace, u_vrai[:,75], linewidth=6, color='b')
        axs[2].plot(xspace, u_t_75, linewidth=6, linestyle='dashed', color='r')
        axs[2].set_title('t=0.75', **title_style2)
        axs[2].set_xlabel('x', fontsize = 15)
        
        fig.legend(handles=(l1,l2),labels=('Exact','Predicted'),loc='upper right', fontsize = 15)
        plt.savefig("plot/25_50_75_"+nom_save)
        return()
        

    def plot_loss_history(self, nom_save='plot_hist_loss'):
        '''
        Plot l'historique de la fonction loss
        '''
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(111)
        

        ax.semilogy(range(len(self.hist_loss)), self.hist_loss,'k-') 

        title_style = dict(pad=10, fontname="Ubuntu", fontsize=18)
        ax.set_xlabel('$n_{epochs}$', fontsize = 15)
        ax.set_ylabel('loss', fontsize = 15)
        ax.set_title('Evolution de loss', **title_style)

        plt.savefig("plot/loss_history_"+nom_save)
        return()
