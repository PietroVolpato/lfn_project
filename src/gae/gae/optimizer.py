import tensorflow as tf


class OptimizerAE:
    def __init__(self, preds, labels, pos_weight, norm, learning_rate):
        # Predizioni e etichette
        preds_sub = preds
        labels_sub = labels

        # Calcolo della funzione di costo
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub,
                labels=labels_sub,
                pos_weight=pos_weight
            )
        )

        # Ottimizzatore
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Funzioni per l'accuratezza
        self.correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
            tf.cast(labels_sub, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def minimize(self, tape):
        """Calcola e applica i gradienti."""
        gradients = tape.gradient(self.cost, self.optimizer.variables())
        self.optimizer.apply_gradients(zip(gradients, self.optimizer.variables()))


class OptimizerVAE:
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, learning_rate):
        # Predizioni e etichette
        preds_sub = preds
        labels_sub = labels

        # Calcolo della funzione di costo (log-likelihood)
        self.log_lik = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub,
                labels=labels_sub,
                pos_weight=pos_weight
            )
        )

        # Perdita latente KL-divergence
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(
            tf.reduce_sum(
                1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.exp(2 * model.z_log_std),
                axis=1
            )
        )

        # Costo totale (ELBO)
        self.cost = self.log_lik - self.kl

        # Ottimizzatore
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Funzioni per l'accuratezza
        self.correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
            tf.cast(labels_sub, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def minimize(self, tape):
        """Calcola e applica i gradienti."""
        gradients = tape.gradient(self.cost, self.optimizer.variables())
        self.optimizer.apply_gradients(zip(gradients, self.optimizer.variables()))
