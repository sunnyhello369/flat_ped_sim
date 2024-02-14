
def _build_net(self):
    with tf.name_scope('inputs'):
        self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
    # fc1
    layer = tf.layers.dense(
        inputs=self.tf_obs,
        units=10,
        activation=tf.nn.tanh,  # tanh activation
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        bias_initializer=tf.constant_initializer(0.1),
        name='fc1'
    )
    # fc2
    all_act = tf.layers.dense(
        inputs=layer,
        units=self.n_actions,
        activation=None,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        bias_initializer=tf.constant_initializer(0.1),
        name='fc2'
    )

    self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

    with tf.name_scope('loss'):
        # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
        # or in this way:
        # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

    with tf.name_scope('train'):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
