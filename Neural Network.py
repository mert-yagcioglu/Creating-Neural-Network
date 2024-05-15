#!/usr/bin/env python
# coding: utf-8

# In[5]:


import random
import math

class NeuralNetwork:
    LEARNING_RATE = 0.5 #Öğrenme değeri 0.5 alınmış. Fark 0.5li arttıralacak veya azaltılacak. Burda 0.5 çok büyük bir değer normalde 10 üzeri -3, -4, -5 civarı olmalıdır. Genelde budur.

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)  #Kaç gizli layer varsa onları ve gizli katmandaki biasleri yazıyor
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

        
#Bu sadece girişten gizli katmana doğru
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1
#Eğer yoksa rastgele ağırlık ataması yap, eğer varsa aynısını yaz, bir sonrakine geç.

                
                
                
#Bu da gizli katmandan çıkışa doğru
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

                
                
#Burda sonuçları kontrol etmek için.
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

        
#Burda ileri besleme yapıyoruz yani ilk iterasyonda                 
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    
    
#İlk ileri gidicez ve hatayı hesaplıycaz daha sonra hatayı dağıtıp dğzeltmek için backpropagation yapıcaz. Forward ve sonra
#Backward yapmamız bir iterasyon demektir.
#Batch öğrenme demek, rastgele 10bin tane eğtitim verisinden sonra ağın ağırlıklarını güncelle
#Online öğrenme demek her eğitim verisinden sonra ağırlıklarını güncelle demektir.
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
#İlk ileri doğru gidiyoruz.
        
    
#Giriş değerlerinden sonraki değerleri hesaplıyor.
        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
#Çıkışta elde ettiğimiz hatayı hesaplıyoruz. Bu hatayı backward'da geriye doğru dağıtıp, düzelteceğiz.
            

    
#Gizli katmanda da 1.adımdaki işlemlerin aynısını yapıyor.
        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            
#Backpropagation'da gizli katman nöronlarımızın çıkışına göre hata türevini hesaplıyoruz.
            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

            
            
#Ağırlıkları, yolları güncelliyoruz.
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

                
#Bunda da gizli nörondaki ağırlıkları hesaplıyoruz.
        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

                
#Burda toplam hatayı hesaplıyoruz. Backward'da geri dağıtıp, ağırlıkları düzeltmek için hesaplıyoruz.
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
#Boş self nöron listesi oluşturuyor ve daha sonra bu boş olan listeye kaç tane nöron gelecekse döngüsel şekilde listeye ekliyor



#Inspect kaç tane nöron olduğunu göstermek için
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):    #Burda o katmandaki tüm nöronların üzerinde geziyor
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)


            
            
            
    def feed_forward(self, inputs):
        outputs = []                            #Çıkışlar için boş liste oluşturmuş
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))        #Çıkışları hesaplayıp bu boş listenin içine eklemiş
        return outputs

    
    
#Sadece çıkışları hesaplayıp çıkış değerlerini bu listeye atıp yazar.
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs
    
    
    
#*************************************************************************************************************************



class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
#Burda bias ve weights degerlerini tanimliyoruz.

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output
#Giriş ve çıkış değerlerini tanımlıyoruz.

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]      #input ve ağırlıkları çarpıp bunları toplaya toplaya gidiyor.
        return total + self.bias


    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))
#Burda logistik fonksiyon olan 1 bölü 1 artı e üzeri toplam girişlerin hesaplamasını yapıp yazıyor

#Pythonda self kendini tanimliyor. Yani tahmin ettigimiz deger degil dogru sonuca self deniyor.




    
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

  
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2


    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)


    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)


    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]



nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# XOR örnek:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))


# In[ ]:




