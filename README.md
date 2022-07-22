![TorchEA Logo](./docs/logo.png)

# torchea
Training and construction of torch models based on evolutionary algorithms

# Introduction

Neural networks have the form of a matrix, the values of which are represented as weights of synaptic connections and are subjected to operations. These weights are selected using the backward error propagation method. However, this concept has a number of limitations:
- The backward error propagation method works only with differentiable activation functions.
- The values of the weights can tend to zero, but never to be zero. This means that the weights become noise, which interferes with the operation of the neural network. Pruning was invented to solve this problem.
- The need to find the right neural network architecture to solve the problem. Finding the right architecture involves trying a huge number of combinations and is limited by the mental capacity of machine learning engineers.
- The time to traverse a neural network depends on the engineer. The engineer tends to make the model more complex in order to increase accuracy, but then the time to traverse the neural network increases. This balance is very difficult to maintain using only the skills of the engineer.
- One-criteria learning. Neural network models are trained with respect to one criterion (error function), ignoring other necessary criteria. Engineers use metrics for training, but they allow only to observe the training, but do not affect the training itself.


Translated with www.DeepL.com/Translator (free version)

Нейронные сети имеют вид матрицы, значения которой представляются как веса синаптических связи и подвергаются операциям. Эти веса подбираются с помощью метода обратного распространения ошибки. Однако такая концепция имеет ряд ограничений:
- Метод обратного распространения ошибки работает только с дифференцируемыми функциями активации.
- Значения весов могут стремиться к нулю, но не когда не быть нулём. Это означает, что веса становятся шумами, которые мешают работе нейронной сети. Для решения этой проблемы был придуман прунинг.
- Необходимость найти правильную архитектуру нейронной сети для решения задачи. Поиск правильной архитектуры подразумевает перебор огромного количества комбинаций и ограничен умственными способностями иженеров по машинному обучению.
- Время прохода через нейронную сеть зависит от инженера. Инженер стремится усложнить модель, чтобы увеличить точность, но тогда увеличивается время прохода через нейронную сеть. Этот баланс очень сложно соблюсти прибегая лишь к умениям инженера.
- Однокритериальное обучение. Модели нейронных сетей обучаются относительно одного критерия(функции ошибки), игнорируя другие необходимые критерии. При обучении инженеры используют метрики, но они позволяют лишь наблюдать за обучением, но не влияют на само обучение.

The goal of this repository is to create tools and algorithms for automatic construction of artificial neural networks based on evolutionary algorithms. Evolutionary algorithms do not require differentiable functions, are able to find the right combination well, the selection of weights and architecture can be represented as a combinatorial problem, and have the ability to use more than one criterion for their work. This repository is built on pytorch and has all its advantages.

Translated with www.DeepL.com/Translator (free version)

Целью данного репозитория является создание инструментов и алгоритмов для автоматического построения искуственных нейронных сетей на основе эволюционных алгоритмов. Эволюционные алгоритмы не требуют дифференцируемых функций, способны хорошо находить нужную комбинацию, подбор весов и архитектуры можно представить как комбинаторную задачу, и имеют возможность использовать более одного критерия для своей работы. Этот репозиторий построен на pytorch и имеет все его преимущества.


# About evolutionary algorithms

Evolutionary algorithms can solve combinatorial problems well. They simulate the mechanism of nature's evolution and can find a good combination of both real and discrete values relative to a given fitness function(s). In doing so, they don't go through all the possible choices, but only a small fraction of the possible choices. This is very similar to what happens to an artificial neural network when it is trained, because training a neural network can be thought of as going through real numbers. I have not found an adequate explanation for why evolutionary algorithms work. 

Translated with www.DeepL.com/Translator (free version)

Эволюционные алгоритмы могут хорошо решать комбинаторные задачи. Они имитируют механизм эволюции природы и могут найти хорошую комбинацию как вещественных, так и дискретных значений относительно заданной функции(функций) приспособленности. При этом они перебирают не всевозможные варианты, а лишь небольшую часть возможных вариантов. Это очень похоже на то, что происходит с искусственной нейронной сетью при ее обучении, ведь обучение нейронной сети можно представить как перебор вещественных чисел. Я не нашел адекватного объяснения почему работают эволюционные алгоритмы. 

Once there were only classical interpretive machine learning methods, people understood everything, but they did not work well. Then ensemble methods and artificial neural networks came along, and people understood them worse, but they worked better. No one understands how evolutionary algorithms work, so they must work even better. :D

Translated with www.DeepL.com/Translator (free version)

Когда-то существовали только классические интерпретируемые методы машинного обучения, люди все понимали, но они плохо работали. Потом появились ансамблевые методы и искусственные нейронные сети, люди стали понимать их хуже, но они работали лучше. Никто не понимает как работают эволюционные алгоритмы, поэтому они должны работать еще лучше. :D

Genetic algorithms and differential evolution are used for learning. To solve the problem, evolutionary algorithms manipulate the traits of individuals. The traits we will consider as activation functions, the presence of connections between neurons, and the weights of these connections.

Translated with www.DeepL.com/Translator (free version)

Для обучения используется генетические алгоритмы и дифференциальная эволюция. Чтобы решить задачу, эволюционные алгоритмы манипулируют признаками особей. Признаками мы будет считать функции активации, наличие связи между нейронами и веса этих связей.