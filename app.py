from flask import Flask, render_template, request, jsonify
import os
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv
import vertexai

load_dotenv()

app = Flask(__name__)

project_id = os.getenv("project_id")
region = os.getenv("region")

try:
    vertexai.init(project=project_id, location=region)
    model = GenerativeModel("gemini-2.5-flash-preview-05-20")
    print("Model initialized successfully")
except Exception as e:
    print(f"Failed to initialize model: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/gemini', methods=['GET', 'POST'])
def vertex_ai():
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form.get('user_input')

    if not user_input:
        return jsonify(error="User input is empty"), 400

    try:
        responses = model.generate_content(user_input, stream=True)
        res = []
        for response in responses:
            try:
                if response.candidates:
                    res.append(response.candidates[0].content.parts[0].text)
                else:
                    print("No candidates in response")
            except IndexError:
                print("Invalid response format")
            except Exception as e:
                print(f"Error processing response: {e}")
        if res:
            final_res = "".join(res)
            return jsonify(content=final_res)
        else:
            return jsonify(error="No response generated"), 500
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)