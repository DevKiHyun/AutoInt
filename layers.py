import tensorflow as tf
import tensorflow.contrib.layers.variance_scaling_initializer as he_initializer


class InteractingLayer:
    """An tensorflow implementation of Interacting Layer used in AutoInt.
        This models the correlations between different feature fields by multi-head self-attention mechanism.
        Arguments
        ---------
            embedding_size: Integer. The size of input feature vector.
            att_embedding_size: Integer. The embedding size in multi-head self-attention network.
            head_num: Integer. The head number in multi-head self-attention network.
            use_res: Boolean. Whether or not use standard residual connections before output.
            dropout_p: Float in [0, 1). Dropout rate of multi-head self-attention.

        Input shape
        -----------
            3D tensor with shape `(batch_size, field_size, embedding_size)`.

        Output shape
        ------------
            3D tensor with shape `(batch_size, field_size, att_embedding_size * head_num)`.

        References
        ----------
            This code is based sakami0000 github code. [https://github.com/sakami0000/recommendation/tree/master/autoint]
            Paper:
            [Song W, Shi C, Xiao Z, et al.
             AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J].
             arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)]
        """

    def _nn_layer(self, inputs, inputs_size, n_hidden, add_bias=True, name=None, activation=None):
        weights_size = [inputs_size, n_hidden]
        layer_weights = tf.Variable(he_initializer()(shape=weights_size), name=name+'weigths', trainable=True)

        hidden_layer = tf.matmul(inputs, layer_weights)

        if add_bias == True:
            layer_bias = tf.Variable(tf.zeros([n_hidden]), name=name+'bias', trainable=True)
            hidden_layer = tf.add(hidden_layer, layer_bias)

        if activation != None:
            hidden_layer = activation(hidden_layer)

        return hidden_layer

    def __init__(self,inputs, embedding_size, att_embedding_size=8, head_num=2, use_res=True, dropout_p=1., name=None):
        # inputs shape : (batch_size, field size, embedding_size)
        if head_num <= 0:
            if head_num <= 0:
                raise ValueError('head_num must be a int > 0')

        self.inputs = inputs
        self.name = name
        self.batch_size = inputs.get_shape().as_list()[0]
        self.embedding_size = embedding_size
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.dropout_p = dropout_p

        # (batch_size, field_size, embedding_size) -> (batch_size, field_size, att_embedding_size * head_num)
        self.w_query = self._nn_layer(inputs, inputs_size=self.embedding_size, n_hidden=self.att_embedding_size * self.head_num, name=self.name+'query')
        self.w_key = self._nn_layer(inputs, inputs_size=self.embedding_size, n_hidden=self.att_embedding_size * self.head_num, name=self.name+'key')
        self.w_value = self._nn_layer(inputs, inputs_size=self.embedding_size, n_hidden=self.att_embedding_size * self.head_num, name=self.name+'value')

        # (batch_size * head_num, field_size, att_embedding_size)
        querys = tf.concat(tf.split(self.w_query, num_or_size_splits=self.att_embedding_size, axis=2), axis=0)
        keys = tf.concat(tf.split(self.w_key, num_or_size_splits=self.att_embedding_size, axis=2), axis=0)
        values = tf.concat(tf.split(self.w_value, num_or_size_splits=self.att_embedding_size, axis=2), axis=0)

        # (batch_size * head_num, field_size, field_size)
        attention = tf.matmul(querys, tf.transpose(keys, perm=[1,2]))
        attention = tf.div(attention, tf.sqrt(self.embedding_size))
        attention = tf.nn.softmax(attention, axis=-1)
        attention = tf.nn.dropout(attention, self.dropout_p)

        # (batch_size * head_num, field_size, att_embedding_size)
        result = tf.matmul(attention, values)
        if self.use_res:
            self.w_res = self._nn_layer(self.inputs, inputs_size=self.embedding_size, n_hidden=self.att_embedding_size * self.head_num, name=self.name+'res')
            result = tf.add(result, self.w_res)

        # (batch_size, field_size, att_embedding_size * head_num)
        result = tf.concat(tf.split(result, num_or_size_splits=self.batch_size, axis=0), axis=2)
        self.layer = tf.nn.relu(result)
