import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dense_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layer_normalization_1 = tf.keras.layers.LayerNormalization()
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype=tf.int64)
        else:
            padding_mask = None

        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layer_normalization_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layer_normalization_2(proj_input + proj_output)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(length)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dense_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask()
        # causal_mask = None

        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype=tf.int64)
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    @staticmethod
    def get_causal_attention_mask(inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, None]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int64)
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.convert_to_tensor([1, 1])], axis=0)
        return tf.tile(mask, mult)


@tf.keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()

        self.start_code = 2

        self.embed_dim = 64
        self.dense_dim = 64
        self.num_heads = 2

        self.vocab_size = vocab_size

        self.seq_in_length = 28
        self.seq_out_length = 27

        # encoder
        self.encoder_positional_embedding = PositionalEmbedding(self.seq_in_length, self.vocab_size, self.embed_dim)
        self.encoder_transformer = TransformerEncoder(self.embed_dim, self.dense_dim, self.num_heads)

        # decoder
        self.decoder_positional_embedding = PositionalEmbedding(self.seq_out_length, self.vocab_size, self.embed_dim)
        self.decoder_transformer = TransformerDecoder(self.embed_dim, self.dense_dim, self.num_heads)
        self.decoder_dropout = tf.keras.layers.Dropout(0.5)
        self.decoder_dense = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, inputs):
        seq_in, seq_out = inputs
        encoder_output = self.encoder(seq_in)
        x = self.decoder(encoder_output, seq_out)
        return x

    def encoder(self, seq_in):
        x = seq_in
        x = self.encoder_positional_embedding(x)
        x = self.encoder_transformer(x)
        return x

    def decoder(self, encoder_output, seq_out):
        x = seq_out
        x = self.decoder_positional_embedding(x)
        x = self.decoder_transformer(x, encoder_output)
        x = self.decoder_dropout(x)
        x = self.decoder_dense(x)
        return x

    def predict_sentence(self, input_sentence):
        seq_in = input_sentence
        encoder_output = self.encoder(seq_in)

        decoded_part = self.start_code * tf.ones((seq_in.shape[0], 1), dtype=tf.int64)

        for i in tf.range(self.seq_out_length):
            predictions = self.decoder(encoder_output, decoded_part)
            next_token = tf.argmax(predictions[:, i, :], axis=1)
            next_token = tf.expand_dims(next_token, axis=1)
            decoded_part = tf.concat([decoded_part, next_token], axis=1)

        return decoded_part.numpy()
