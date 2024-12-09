from flask import Flask, request, render_template, jsonify
import openai
import os

# Initialize Flask app
app = Flask(__name__)

# Set up OpenAI API Key (use environment variables for security)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Routes
@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint to summarize text."""
    try:
        # Get input text from the form
        input_text = request.form.get('text')

        if not input_text:
            return jsonify({'error': 'No text provided!'}), 400

        # Call OpenAI API to summarize the text
        summary = call_openai_summarize(input_text)

        return jsonify({'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    """Endpoint to translate summarized text."""
    try:
        summary = request.form.get('summary')
        language = request.form.get('language')

        if not summary or not language:
            return jsonify({'error': 'Missing summary or language!'}), 400

        # Call OpenAI API to translate the text
        translated_text = call_openai_translate(summary, language)

        return jsonify({'translation': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper Functions
def call_openai_summarize(text):
    """Summarizes text using OpenAI."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following text:\n{text}"}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def call_openai_translate(text, language):
    """Translates text into the specified language using OpenAI."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that translates text into {language}."},
            {"role": "user", "content": f"Translate this text into {language}:\n{text}"}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

if __name__ == '__main__':
    app.run(debug=True)
