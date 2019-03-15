import tensorflow as tf
import tensorflow.contrib.layers.variance_scaling_initializer as he_initializer

from layers import InteractingLayer

class AutoInt:
    """Tensorflow implementation of AutoInt.
        Arguments
        ---------
            feature_size = Integer. Denote n. The Dimension of concatenated features
            num_embeddings: Integer. Denote M. The number of input features or the number of total feature field.
            embedding_dim: Integer. Denote d, The size of input feature vector or the size of the feature embedding.
            att_layer_num: Integer. The Interacting Layer number to be used.
            att_embedding_size: Integer. The embedding size in multi-head self-attention network.
            att_head_num: Integer. The head number in multi-head self-attention network.
            att_use_res: Boolean. Whether or not use standard residual connections before output.
            att_dropout_p: Float in [0, 1). Dropout rate of multi-head self-attention.
            activation: {'sigmoid', 'linear'}. Output activation.
        References
        ----------
            This code is based sakami0000 github code. [https://github.com/sakami0000/recommendation/tree/master/autoint]
            [Song W, Shi C, Xiao Z, et al.
             AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J].
             arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
        """

    def __init__(self, feature_size, num_embeddings, embedding_dim, output_dim,
                 att_layer_num=3, att_embedding_size=8, att_head_num=2,
                 att_use_res=True, att_dropout_p=1.):

        self.feature_size = feature_size  # 'n' size
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")  # batch_size * M
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")  # batch_size * M
        self.att_layer_num = att_layer_num
        self.output_dim = output_dim

        self.embedding_weights = tf.Variable(he_initializer()(shape=[self.feature_size, embedding_dim]), name="feature_embeddings")
        self.embedding = tf.nn.embedding_lookup(self.embedding_weights, self.feat_index)  # batch_size * M * d
        feat_value = tf.reshape(self.feat_value, shape=[-1, num_embeddings, 1])
        self.embedding = tf.mutiply(self.embedding, feat_value)  # batch_size * M * d
        self.embedding = tf.nn.dropout(self.embedding, att_dropout_p)  # batch_size * M * d

        for i in range(self.att_layer_num):
            if i == 0:
                inputs = self.embedding
                input_dim = embedding_dim
            else:
                inputs = interact_layer
                input_dim = att_embedding_size * att_head_num

            interact_layer = InteractingLayer(inputs,
                                              embedding_size=input_dim,
                                              att_embedding_size=att_embedding_size,
                                              head_num=att_head_num,
                                              use_res=att_use_res,
                                              dropout_p=att_dropout_p,
                                              name='interacting_layer_{}'.format(i+1)).layer

        self.flatten = tf.reshape(interact_layer, shape=[-1, att_embedding_size * att_head_num])

        # Make output layer to attention
        output_weights = tf.Variable(he_initializer()(shape=[att_embedding_size * att_head_num, self.output_dim]), name="feature_embeddings")
        self.output = tf.matmul(self.flatten, output_weights)
        self.output = tf.nn.sigmoid(self.output)
