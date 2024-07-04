from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout

class TextCNN(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 kernel_sizes=[3, 4, 5],
                 class_num=3,
                 last_activation='softmax'):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims)
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(Conv1D(128, kernel_size, activation='relu'))
            self.max_poolings.append(GlobalMaxPooling1D())
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        # Embedding layer
        embedding = self.embedding(inputs)
        # Convolutional Layers
        convs = []
        for i in range(len(self.kernel_sizes)):
            c = self.convs[i](embedding)
            # Max-Pooling Layers
            c = self.max_poolings[i](c)
            convs.append(c)
        # concatenate max pooling layers
        x = Concatenate()(convs)
        # Fully Connected Layer to predict
        output = self.classifier(x)
        return output


'''
stride: the step the filter moves each time, 1 stride, 2 stride (skip some details)...
epoch: one iteration of training

batch: default batch is to use all the data for training, can be slow
mini batch: only train on a small subset of training data for each epoch
batch size: a hyperparameter
'''