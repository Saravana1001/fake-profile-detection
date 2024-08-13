from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/xgb_model.pkl')

# Replace these with the min and max values from your dataset
MIN_FOLLOWERS = 0
MAX_FOLLOWERS = 10000
MIN_FOLLOWING = 0
MAX_FOLLOWING = 5000


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    proba = None
    
    if request.method == 'POST':
        # Extract raw data from form
        followers = int(request.form['followers'])
        following = int(request.form['following'])

        # Calculate normalized values
        edge_followed_by = (followers - MIN_FOLLOWERS) / (MAX_FOLLOWERS - MIN_FOLLOWERS)
        edge_follow = (following - MIN_FOLLOWING) / (MAX_FOLLOWING - MIN_FOLLOWING)

        # Extract other data from form
        data = {
            'edge_followed_by': [edge_followed_by],
            'edge_follow': [edge_follow],
            'username_length': [int(request.form['username_length'])],
            'username_has_number': [int(request.form['username_has_number'])],
            'full_name_has_number': [int(request.form['full_name_has_number'])],
            'full_name_length': [int(request.form['full_name_length'])],
            'is_private': [int(request.form['is_private'])],
            'is_joined_recently': [int(request.form['is_joined_recently'])],
            'has_channel': [int(request.form['has_channel'])],
            'is_business_account': [int(request.form['is_business_account'])],
            'has_guides': [int(request.form['has_guides'])],
            'has_external_url': [int(request.form['has_external_url'])],
        }
        
        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
    return render_template(
        'index.html',
        prediction=prediction,
        proba=proba,
        min_followers=MIN_FOLLOWERS,
        max_followers=MAX_FOLLOWERS,
        min_following=MIN_FOLLOWING,
        max_following=MAX_FOLLOWING
    )

if __name__ == '__main__':
    app.run(debug=True)
