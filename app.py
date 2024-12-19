import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objs as go

# Page config
st.set_page_config(page_title="UCL Predictions", layout="wide")

# Load the saved model
with open('ucl_model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Team logos mapping
team_logos = {
    'Real Madrid': '/static/logos/real_madrid.png',
    'Barcelona': '/static/logos/barcelona.png',
    'Liverpool' : '/static/logos/liverpool.png',
    'Aston Villa' : '/static/logos/aston_villa.png',
    'Inter Milan' : '/static/logos/inter_milan.png',
    'Arsenal' : '/static/logos/arsenal.png',
    'Bayer Leverkusen' : '/static/logos/bayer_leverkusen.png',
    'Brest' : '/static/logos/brest.png',
    'Lille' : '/static/logos/lille.png',
    'Bayern Munich'	: '/static/logos/bayern_munchen.png',
    'Atletico Madrid' : '/static/logos/atletico.png',
    'Borussia Dortmund' : '/static/logos/borussia_dortmund.png',
    'AC Milan' : '/static/logos/ac_milan.png',
    'Atalanta' : '/static/logos/atalanta.png',
    'Benfica' : '/static/logos/benfica.png',
    'Juventus' : '/static/logos/juventus.png',
    'Monaco' : '/static/logos/monaco.png',
    'Sporting Lisbon'	: '/static/logos/sporting.png',
    'Feyenoord'	: '/static/logos/feyenoord.png',
    'Club Brugge' : '/static/logos/club_brugge.png',
    'Celtic' :  '/static/logos/celtic.png',
    'Manchester City' :  '/static/logos/man_city.png',
    'PSV Eindhoven' :   '/static/logos/psv.png',
    'Dinamo Zagreb' :   '/static/logos/dinamo_zaghreb.png',
    'Paris Saint-Germain' :   '/static/logos/psg.png',
    'Stuttgart' :   '/static/logos/stuttgart.png',
    'Sparta Prague' :   '/static/logos/sparta_prague.png',
    'Shakhtar Donetsk' :   '/static/logos/shakhtar_donetsk.png',
    'Sturm Graz' :   '/static/logos/sturm_graz.png',
    'Girona' :   '/static/logos/girona.png',
    'Crvena Zvezda' :   '/static/logos/crvena_zvezda.png',
    'Red Bull Salzburg' :   '/static/logos/salzburg.png',
    'Bologna' :   '/static/logos/bologna.png',
    'RB Leipzig' :   '/static/logos/leipzig.png',
    'BSC Young Boys Bern' :   '/static/logos/young_boys.png',
    'SK Slovan Bratislava' :   '/static/logos/slovan_bratislava.png'
}

def prepare_prediction_data(df):
    """
    Prepare data for predictions
    """
    return df

# Load and prepare data
df = pd.read_csv('teams_details.csv')
X = prepare_prediction_data(df)
predictions = model.predict(X)

# Create results dataframe
results_df = pd.DataFrame({
    'Team': df['Team'],
    'Progression_Probability': predictions,
    'Opponent_Strength': df['Opponent_Strength'],
    'Goal_Efficiency': df['Goal_Efficiency']
})

# Sort and add position
results_df = results_df.sort_values(by='Progression_Probability', ascending=False)
results_df['Position'] = range(1, len(results_df) + 1)
results_df['Logo'] = results_df['Team'].map(team_logos).fillna('/static/logos/default.png')

# Title
st.title('UEFA Champions League Round of 16 Predictions')

# Create and display plotly chart
fig = go.Figure(data=[
    go.Bar(
        x=results_df['Team'],
        y=results_df['Progression_Probability'],
        text=[f'{pred:.4f}' for pred in results_df['Progression_Probability']],
        textposition='auto',
        marker=dict(
            color=results_df['Progression_Probability'],
            colorscale='Viridis',
            showscale=True
        )
    )
])

fig.update_layout(
    xaxis=dict(title='Team', tickangle=45),
    yaxis=dict(title='Progression Probability', tickformat='.4f'),
    height=600,
    margin=dict(b=100, l=50, r=50, t=50)
)

st.plotly_chart(fig, use_container_width=True)

# Display table with tooltips
st.write("### Detailed Predictions")

# Add column descriptions
st.markdown("""
    ##### Column Descriptions:
    - **Position**: Team's ranking based on predicted progression probability
    - **Team**: Club name and logo
    - **Progression Probability**: Predicted likelihood of advancing (0-1 scale)
    - **Opponent Strength**: Measure of opposition quality
    - **Goal Efficiency**: Team's scoring effectiveness
""")

# Create the dataframe display
st.dataframe(
    results_df[['Position', 'Team', 'Progression_Probability', 'Opponent_Strength', 'Goal_Efficiency']],
    column_config={
        'Position': st.column_config.NumberColumn('Position', format='%d'),
        'Team': 'Team',
        'Progression_Probability': st.column_config.NumberColumn('Progression Probability', format='%.6f'),
        'Opponent_Strength': st.column_config.NumberColumn('Opponent Strength', format='%.2f'),
        'Goal_Efficiency': st.column_config.NumberColumn('Goal Efficiency', format='%.2f')
    },
    hide_index=True
)

# Add disclaimer
st.warning("""
    ⚠️ **DISCLAIMER**
    
    This predictive model is for analytical and educational purposes only. It should NOT be used as a basis for sports betting or gambling. 
    
    Predictions are based on historical data and statistical models, and do not guarantee future outcomes. Sports events are inherently unpredictable.
""")