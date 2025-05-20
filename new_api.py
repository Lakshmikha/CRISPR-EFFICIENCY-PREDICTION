from flask import Flask, request, jsonify
from original_ml_code import sequence_to_one_hot, target_gene_to_one_hot, predict_efficiency
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

MIN_POS = 1.0 #na
MAX_POS = 2826.0 #na

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    mrna_sequence = data['mrnaSequence'].upper()
    gene_name=data.get(('targetGene'))
    
    gene_to_index={
    "CCDC101" : 0 , "CD13": 1 , "CD15": 2, "CD28" : 3, "CD33" : 4,
    "CD43" : 5, "CD45" : 6, "CD5" : 7, "CUL3" : 8, "H2-K" : 9,
    "HPRT1" : 10, "MED12" : 11, "NF1" : 12, "NF2" : 13, "TADA1" :14,
    "TADA2B" : 15, "THY1" : 16

    }
    target_gene_index = gene_to_index.get(gene_name)
    if target_gene_index is None:
        return jsonify({'error': 'Invalid target gene name provided.'}), 400

    cut_position = int(data['cutPosition'])

    # convert inputs
    mrna_one_hot = sequence_to_one_hot(mrna_sequence)
    gene_one_hot = target_gene_to_one_hot(target_gene_index)
    normalized_cut_position = (cut_position - MIN_POS) / (MAX_POS - MIN_POS) #na

    # predict
    efficiency_score = predict_efficiency(mrna_one_hot, gene_one_hot, normalized_cut_position) #na

    # build a small “nearby” list (±2 positions)
    nearby = []
    for pos in range(cut_position - 2, cut_position + 3): #na
        if pos<1: #NA
            continue #NA
        norm_pos = (pos-MIN_POS)/(MAX_POS-MIN_POS) #na
        eff = predict_efficiency(mrna_one_hot, gene_one_hot, norm_pos) #na        
        nearby.append({
            'position': pos,
            'efficiency': round(eff, 6)
        })

    return jsonify({
        'type':       'single',
        'position':   cut_position,
        'efficiency': round(efficiency_score, 6),
        'nearby':     nearby
    })


if __name__ == '__main__':
    app.run(debug=True)
