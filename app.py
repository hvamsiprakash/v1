import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Load the model weights from the URL
model_url = "https://github.com/hvamsiprakash/v1/raw/main/model_weights.h5"
model_path = "image_captioning_model_weights.h5"

# Download the model weights if not already present
if not os.path.exists(model_path):
    st.write("Downloading model weights...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    st.write("Model weights downloaded.")

# Load the model architecture and weights
@st.cache(allow_output_mutation=True)
def load_model():
    # Define the model architecture (same as in your training script)
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=2)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=3)
    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    
    # Load the weights
    caption_model.load_weights(model_path)
    return caption_model

# Define the model components (same as in your training script)
def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.0)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=None, training=training)
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.cross_attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=combined_mask, training=training)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        cross_attention_output_2 = self.cross_attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask, training=training)
        out_2 = self.layernorm_2(out_1 + cross_attention_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def call(self, inputs, training=False):
        batch_img, batch_seq = inputs
        if self.image_aug:
            batch_img = self.image_aug(batch_img)
        img_embed = self.cnn_model(batch_img, training=training)
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        return batch_seq_pred

# Constants
IMAGE_SIZE = (299, 299)
SEQ_LENGTH = 24
VOCAB_SIZE = 13000
EMBED_DIM = 512
FF_DIM = 512

# Load the model
caption_model = load_model()

# Streamlit app
st.title("Image Captioning with Streamlit")
st.write("Upload an image and the model will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating caption...")

    # Preprocess the image
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Generate caption
    def greedy_algorithm(image):
        image = tf.expand_dims(image, 0)
        image = caption_model.cnn_model(image)
        encoded_img = caption_model.encoder(image, training=False)
        decoded_caption = "<start> "
        for i in range(SEQ_LENGTH - 1):
            tokenized_caption = vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = INDEX_TO_WORD[sampled_token_index]
            if sampled_token == "<end>":
                break
            decoded_caption += " " + sampled_token
        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        return decoded_caption

    # Get the vocabulary
    vocab = vectorization.get_vocabulary()
    INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocab)}

    # Generate the caption
    caption = greedy_algorithm(image)
    st.write("Generated Caption:", caption)
