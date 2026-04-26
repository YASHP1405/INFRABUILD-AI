from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
# Enable CORS for all routes, allowing your frontend to securely fetch data
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to verify the server is running."""
    return jsonify({
        "status": "success",
        "message": "InfraBuild AI Automation API is up and running!"
    }), 200

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint returning JSON response to verify Frontend connection."""
    return jsonify({
        "status": "success",
        "data": {
            "platform": "InfraBuild AI",
            "version": "1.0.0",
            "message": "Backend to Frontend connection successful!"
        }
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Placeholder AI processing endpoint."""
    try:
        data = request.get_json()
        input_text = data.get('text', '') if data else ''
        
        if not input_text:
            return jsonify({"status": "error", "message": "No input text provided."}), 400
            
        # Simulated AI processing logic
        result = {
            "original_text": input_text,
            "analysis": f"AI Engine processed: '{input_text}'. Pattern recognized successfully.",
            "confidence_score": 0.98
        }
        
        return jsonify({"status": "success", "result": result}), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Render assigns a dynamic port via the PORT env variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
