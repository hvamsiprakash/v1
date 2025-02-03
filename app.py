import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import requests

# Custom layer definitions for Transformer Encoder, Decoder, and Positional Embedding

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.dense_1 = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
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


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=512, sequence_length=25, vocab_size=10000
        )
        self.out = tf.keras.layers.Dense(10000, activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

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
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


# Function to download model weights from GitHub
def download_model_weights():
    model_url = "https://github.com/hvamsiprakash/v1/raw/main/model_weights.h5"
    model_weights_path = "best_model_weights.h5"
    
    if not os.path.exists(model_weights_path):
        response = requests.get(model_url)
        with open(model_weights_path, 'wb') as f:
            f.write(response.content)
    
    return model_weights_path


# Load Model (cached to avoid reloading)
@st.cache_resource
def load_model():
    # Download model weights if not already available
    model_weights_path = download_model_weights()
    
    # Load the model architecture and weights
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=512, dense_dim=512, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=512, ff_dim=512, num_heads=2)
    model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=None
    )

    # Load the pre-trained weights into the model
    model.load_weights(model_weights_path)
    return model

# Load the model
model = load_model()

# Load the vocabulary for text generation
vectorization = keras.layers.TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=25)
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))

# Preprocess image for the model
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to model's input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to generate caption
def generate_caption(image):
    image = preprocess_image(image)

    # Extract features
    img_embed = model.cnn_model(image)
    encoded_img = model.encoder(img_embed, training=False)

    # Generate caption using Transformer decoder
    decoded_caption = "<start> "
    max_length = 24  # Max caption length

    for i in range(max_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]

        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("<start> ", "").strip()
# Streamlit UI

# Set the title of the web app
st.title("🖼️ Image Caption Generator")

# Description text
st.write("""
    Upload an image, and the model will generate a caption for it.
    The caption is generated using a Transformer-based model trained on the Flickr30k dataset.
""")

# Upload image functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Button to generate the caption
    if st.button("Generate Caption"):
        # Call the generate_caption function
        caption = generate_caption(image)
        
        # Display the generated caption
        st.subheader("Generated Caption:")
        st.write(caption)

# Optionally: Show some example images for demonstration
st.subheader("Example Images")

# List of example image URLs
example_images = [
    "https://images.unsplash.com/photo-1519681393781-d3a2c9c6b65b",
    "https://images.unsplash.com/photo-1574158622688-01064c5ac7bb",
    "https://images.unsplash.com/photo-1589152425047-e639b10c1b0b"
]

# Display example images with generated captions
for img_url in example_images:
    img = Image.open(tf.keras.utils.get_file(img_url.split("/")[-1], img_url))
    st.image(img, caption="Example Image", use_column_width=True)
    caption = generate_caption(img)
    st.write(f"**Generated Caption:** {caption}")
