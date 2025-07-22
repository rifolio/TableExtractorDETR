from pathlib import Path
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from pdf_reader import PDFReader
import ai_inference

app = Flask(__name__)
CORS(app)  # allow frontend ↔ backend
logging.basicConfig(level=logging.INFO)


def clean_table(ai_table):
    """
    Converts one AI table (list of rows of mixed types) into a
    list of rows of strings.
    """
    cleaned = []
    for row in ai_table:
        cleaned_row = []
        for cell in row:
            if isinstance(cell, dict) and 'text' in cell:
                cleaned_row.append(cell['text'])
            else:
                cleaned_row.append(str(cell))
        cleaned.append(cleaned_row)
    return cleaned


def process_ai_output(ai_output):
    try:
        # Handle dict with "data" key
        if isinstance(ai_output, dict):
            if 'data' in ai_output and isinstance(ai_output['data'], list) and ai_output['data']:
                # pick only items that have a 'table'
                tables = [item['table'] for item in ai_output['data']
                          if isinstance(item, dict) and 'table' in item]
            # Handle dict with "table" directly
            elif 'table' in ai_output:
                tables = [ai_output['table']]
            else:
                raise ValueError("Missing 'data' or 'table' keys")

        # Handle list of dicts
        elif isinstance(ai_output, list):
            if all(isinstance(item, dict) and 'table' in item for item in ai_output):
                tables = [item['table'] for item in ai_output]
            else:
                raise ValueError("List items are not all dicts with 'table'")

        else:
            raise ValueError(f"Unexpected type: {type(ai_output)}")

        if not tables:
            raise ValueError("No tables found in AI output")

        # Take the first table and clean it
        first_table = tables[0]
        cleaned = clean_table(first_table)
        logging.info(f"Cleaned table: {cleaned}")
        return cleaned

    except ValueError as ve:
        logging.error(f"AI returned unexpected output: {ai_output}")
        return {'error': 'Invalid AI output format', 'details': str(ve)}

    except Exception as e:
        logging.exception("Error processing AI output")
        return {'error': 'Internal server error', 'details': str(e)}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/process', methods=['POST'])
def process():
    # Check upload
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    pdf_file = request.files['pdf']
    temp_path = Path("/tmp") / pdf_file.filename
    pdf_file.save(temp_path)

    try:
        # Convert PDF → images → AI tables
        reader = PDFReader(str(temp_path))
        images = reader.convert_to_images()
        logging.info(f"Converted '{temp_path}' into {len(images)} images.")

        raw_ai = ai_inference.extract_tables_from_images([str(p) for p in images])
        logging.info(f"AI raw output: {raw_ai}")

        # Validate / clean
        processed = process_ai_output(raw_ai)
        if isinstance(processed, dict) and 'error' in processed:
            # structured error
            return jsonify(processed), 500

        return jsonify({'data': processed}), 200

    except Exception as e:
        logging.exception("Server error during /process")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)