import streamlit as st
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch import nn

class CFraud(nn.Module):
    def __init__(self, layers_sz, in_sz, out_sz):
        super(CFraud, self).__init__()
        layers = []
        for sz in layers_sz:
            layers.append(nn.Linear(in_sz, sz))
            in_sz = sz
        self.linears = nn.ModuleList(layers)
        self.out = nn.Linear(layers_sz[-1], out_sz)
        self.act_func = nn.ReLU()
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.linears:
            x = self.act_func(layer(x))
        x = self.output_activation(self.out(x))
        return x

CATEGORY_LIST = ('misc_net', 'grocery_pos', 'entertainment', 'gas_transport',
                'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos',
                'food_dining', 'personal_care', 'health_fitness', 'travel',
                'kids_pets', 'home')

STATE_LIST = ('NC', 'WA', 'ID', 'MT', 'VA', 'PA', 'KS', 'TN', 'IA', 'WV', 'FL',
                'CA', 'NM', 'NJ', 'OK', 'IN', 'MA', 'TX', 'WI', 'MI', 'WY', 'HI',
                'NE', 'OR', 'LA', 'DC', 'KY', 'NY', 'MS', 'UT', 'AL', 'AR', 'MD',
                'GA', 'ME', 'AZ', 'MN', 'OH', 'CO', 'VT', 'MO', 'SC', 'NV', 'IL',
                'NH', 'SD', 'AK', 'ND', 'CT', 'RI', 'DE')

JOB_LIST = ('Psychologist, counselling', 'Special educational needs teacher',
                'Nature conservation officer', 'Patent attorney',
                'Dance movement psychotherapist', 'Transport planner',
                'Arboriculturist', 'Designer, multimedia',
                'Public affairs consultant', 'Pathologist', 'IT trainer',
                'Systems developer', 'Engineer, land', 'Systems analyst',
                'Naval architect', 'Radiographer, diagnostic',
                'Programme researcher, broadcasting/film/video', 'Energy engineer',
                'Event organiser', 'Operational researcher', 'Market researcher',
                'Probation officer', 'Leisure centre manager',
                'Corporate investment banker', 'Therapist, occupational',
                'Call centre manager', 'Police officer',
                'Education officer, museum', 'Physiotherapist', 'Network engineer',
                'Forensic psychologist', 'Geochemist',
                'Armed forces training and education officer',
                'Designer, furniture', 'Optician, dispensing',
                'Psychologist, forensic', 'Librarian, public', 'Fine artist',
                'Scientist, research (maths)', 'Research officer, trade union',
                'Tourism officer', 'Human resources officer', 'Surveyor, minerals',
                'Applications developer', 'Video editor', 'Curator',
                'Research officer, political party', 'Engineer, mining',
                'Education officer, community', 'Physicist, medical',
                'Amenity horticulturist', 'Electrical engineer',
                'Television camera operator', 'Higher education careers adviser',
                'Ambulance person', 'Dealer', 'Paediatric nurse',
                'Trading standards officer', 'Engineer, technical sales',
                'Designer, jewellery', 'Clinical biochemist',
                'Engineer, electronics', 'Water engineer', 'Science writer',
                'Film/video editor', 'Solicitor, Scotland',
                'Product/process development scientist', 'Tree surgeon',
                'Careers information officer', 'Geologist, engineering',
                'Counsellor', 'Freight forwarder',
                'Senior tax professional/tax inspector',
                'Engineer, broadcasting (operations)',
                'English as a second language teacher', 'Economist',
                'Child psychotherapist', 'Claims inspector/assessor',
                'Tourist information centre manager',
                'Exhibitions officer, museum/gallery', 'Location manager',
                'Engineer, biomedical', 'Research scientist (physical sciences)',
                'Purchasing manager', 'Editor, magazine features',
                'Operations geologist', 'Interpreter', 'Engineering geologist',
                'Agricultural consultant', 'Paramedic', 'Financial adviser',
                'Administrator, education', 'Educational psychologist',
                'Financial trader', 'Audiological scientist',
                'Scientist, audiological',
                'Administrator, charities/voluntary organisations',
                'Health service manager', 'Retail merchandiser',
                'Telecommunications researcher', 'Exercise physiologist',
                'Accounting technician', 'Product designer',
                'Waste management officer', 'Mining engineer', 'Surgeon',
                'Therapist, horticultural', 'Environmental consultant',
                'Broadcast presenter', 'Producer, radio',
                'Engineer, communications',
                'Historic buildings inspector/conservation officer',
                'Teacher, English as a foreign language', 'Materials engineer',
                'Health visitor', 'Medical secretary', 'Theatre director',
                'Technical brewer', 'Land/geomatics surveyor',
                'Engineer, structural', 'Diagnostic radiographer',
                'Television production assistant', 'Medical sales representative',
                'Building control surveyor', 'Therapist, sports',
                'Structural engineer', 'Commercial/residential surveyor',
                'Database administrator', 'Exhibition designer',
                'Training and development officer', 'Mechanical engineer',
                'Medical physicist', 'Administrator', 'Mudlogger',
                'Fisheries officer', 'Conservator, museum/gallery',
                'Programmer, multimedia', 'Cytogeneticist',
                'Multimedia programmer', 'Counselling psychologist', 'Chiropodist',
                'Teacher, early years/pre', 'Cartographer', 'Pensions consultant',
                'Primary school teacher', 'Electronics engineer',
                'Museum/gallery exhibitions officer', 'Air broker',
                'Chemical engineer', 'Advertising account executive',
                'Advertising account planner',
                'Chartered legal executive (England and Wales)',
                'Psychiatric nurse', 'Secondary school teacher',
                'Librarian, academic', 'Embryologist, clinical', 'Immunologist',
                'Television floor manager', 'Contractor', 'Health physicist',
                'Copy', 'Bookseller', 'Land', 'Chartered loss adjuster',
                'Occupational psychologist', 'Facilities manager',
                'Further education lecturer', 'Archivist', 'Investment analyst',
                'Engineer, building services', 'Psychologist, sport and exercise',
                'Journalist, newspaper', 'Doctor, hospital', 'Phytotherapist',
                'Pharmacologist', 'Horticultural therapist', 'Hydrologist',
                'Community arts worker', 'Public house manager', 'Architect',
                'Lexicographer', 'Psychotherapist, child',
                'Teacher, secondary school', 'Toxicologist',
                'Commercial horticulturist', 'Podiatrist', 'Building surveyor',
                'Architectural technologist', 'Editor, film/video',
                'Social researcher', 'Wellsite geologist', 'Minerals surveyor',
                'Designer, ceramics/pottery', 'Mental health nurse',
                'Volunteer coordinator', 'Chief Technology Officer',
                'Camera operator', 'Copywriter, advertising', 'Surveyor, mining',
                'Product manager', "Nurse, children's", 'Pension scheme manager',
                'Archaeologist', 'Sub', 'Designer, interior/spatial',
                'Futures trader', 'Chief Financial Officer',
                'Museum education officer', 'Quantity surveyor',
                'Physiological scientist', 'Loss adjuster, chartered',
                'Pilot, airline', 'Production assistant, radio',
                'Immigration officer', 'Retail banker',
                'Health and safety adviser', 'Teacher, special educational needs',
                'Jewellery designer', 'Community pharmacist',
                'Control and instrumentation engineer', 'Make',
                'Early years teacher', 'Sales professional, IT',
                'Scientist, marine', 'Intelligence analyst',
                'Clinical research associate', 'Administrator, local government',
                'Barrister', 'Engineer, control and instrumentation',
                'Clothing/textile technologist', 'Development worker, community',
                'Art therapist', 'Sales executive',
                'Armed forces logistics/support/administrative officer',
                'Optometrist', 'Insurance underwriter', 'Charity officer',
                'Civil Service fast streamer', 'Retail buyer',
                'Magazine features editor', 'Equities trader',
                'Trade mark attorney', 'Research scientist (life sciences)',
                'Psychotherapist', 'Pharmacist, community', 'Risk analyst',
                'Engineer, maintenance', 'Logistics and distribution manager',
                'Water quality scientist', 'Lecturer, further education',
                'Production assistant, television', 'Tour manager',
                'Music therapist', 'Surveyor, land/geomatics',
                'Engineer, production', 'Acupuncturist', 'Hospital doctor',
                'Teacher, primary school', 'Accountant, chartered public finance',
                'Illustrator', 'Scientist, physiological', 'Buyer, industrial',
                'Scientist, research (physical sciences)', 'Radio producer',
                'Manufacturing engineer', 'Animal technologist',
                'Production engineer', 'Biochemist, clinical',
                'Engineer, manufacturing', 'Comptroller',
                'General practice doctor', 'Designer, industrial/product',
                'Prison officer', 'Merchandiser, retail', 'Engineer, drilling',
                'Engineer, petroleum', 'Cabin crew', 'Commissioning editor',
                'Accountant, chartered certified', 'Local government officer',
                'Professor Emeritus', 'Press sub',
                'Chartered public finance accountant', 'Writer',
                'Chief Executive Officer', 'Occupational hygienist',
                'Doctor, general practice', 'Community education officer',
                'Landscape architect', 'Occupational therapist',
                'Special effects artist', 'Civil engineer, contracting',
                "Barrister's clerk", 'Travel agency manager',
                'Associate Professor', 'Neurosurgeon', 'Plant breeder/geneticist',
                'Radio broadcast assistant', 'Field seismologist',
                'Industrial/product designer', 'Metallurgist',
                "Politician's assistant", 'Insurance claims handler',
                'Theme park manager', 'Gaffer', 'Chief Strategy Officer',
                'Heritage manager', 'Ceramics designer', 'Animator',
                'Oceanographer', 'Colour technologist', 'Engineer, agricultural',
                'Therapist, drama', 'Orthoptist', 'Learning mentor',
                'Arts development officer', 'Biomedical engineer',
                'Race relations officer', 'Therapist, music', 'Retail manager',
                'Furniture designer', 'Building services engineer',
                'Maintenance engineer', 'Aid worker', 'Editor, commissioning',
                'Private music teacher', 'Scientist, biomedical',
                'Public relations account executive', 'Dispensing optician',
                'Advice worker', 'Hydrographic surveyor', 'Geoscientist',
                'Environmental health practitioner', 'Learning disability nurse',
                'Chief Operating Officer', 'Scientific laboratory technician',
                'Records manager', 'Barista', 'Marketing executive',
                'Tax inspector', 'Musician', 'Therapist, art',
                'Engineer, automotive', 'Clinical psychologist', 'Warden/ranger',
                'Surveyor, rural practice', 'Sport and exercise psychologist',
                'Education administrator', 'Chief of Staff',
                'Nurse, mental health', 'Music tutor',
                'Planning and development surveyor',
                'Teaching laboratory technician', 'Chief Marketing Officer',
                'Theatre manager', 'Quarry manager',
                'Interior and spatial designer', 'Lecturer, higher education',
                'Regulatory affairs officer', 'Secretary/administrator',
                'Chemist, analytical', 'Designer, exhibition/display',
                'Pharmacist, hospital', 'Site engineer',
                'Equality and diversity officer', 'Public librarian',
                'Town planner', 'Chartered accountant', 'Programmer, applications',
                'Manufacturing systems engineer', 'Web designer',
                'Community development worker', 'Animal nutritionist',
                'Petroleum engineer', 'Information systems manager',
                'Press photographer', 'Insurance risk surveyor', 'Soil scientist',
                'Buyer, retail', 'Public relations officer',
                'Health promotion specialist', 'Psychiatrist',
                'Visual merchandiser', 'Rural practice surveyor', 'Hotel manager',
                'Communications engineer', 'Insurance broker',
                'Radiographer, therapeutic', 'Set designer', 'Tax adviser',
                'Drilling engineer', 'Fitness centre manager', 'Farm manager',
                'Management consultant', 'Energy manager',
                'Museum/gallery conservator', 'Herbalist', 'Osteopath',
                'Statistician', 'Hospital pharmacist', 'Estate manager/land agent',
                'Sports development officer', 'Investment banker, corporate',
                'Biomedical scientist', 'Television/film/video producer',
                'Nutritional therapist', 'Company secretary', 'Production manager',
                'Magazine journalist', 'Media buyer', 'Data scientist',
                'Engineer, civil (contracting)', 'Herpetologist',
                'Garment/textile technologist', 'Scientist, research (medical)',
                'Civil Service administrator', 'Airline pilot', 'Textile designer',
                'Environmental manager', 'Furniture conservator/restorer',
                'Horticultural consultant', 'Firefighter',
                'Geophysicist/field seismologist', 'Psychologist, clinical',
                'Development worker, international aid', 'Sports administrator',
                'IT consultant', 'Presenter, broadcasting',
                'Outdoor activities/education manager', 'Field trials officer',
                'Social research officer, government',
                'English as a foreign language teacher',
                'Restaurant manager, fast food', 'Hydrogeologist',
                'Research scientist (medical)', 'Designer, television/film set',
                'Geneticist, molecular', 'Designer, textile',
                'Licensed conveyancer', 'Emergency planning/management officer',
                'Geologist, wellsite', 'Air cabin crew', 'Seismic interpreter',
                'Surveyor, hydrographic', 'Charity fundraiser', 'Stage manager',
                'Aeronautical engineer', 'Glass blower/designer', 'Ecologist',
                'Horticulturist, commercial', 'Research scientist (maths)',
                'Engineer, aeronautical',
                'Conservation officer, historic buildings', 'Art gallery manager',
                'Advertising copywriter', 'Engineer, civil (consulting)',
                'Oncologist', 'Engineer, materials',
                'Scientist, clinical (histocompatibility and immunogenetics)',
                'Investment banker, operational', 'Medical technical officer',
                'Academic librarian', 'Artist', 'Clinical cytogeneticist',
                'TEFL teacher', 'Administrator, arts', 'Teacher, adult education',
                'Catering manager', 'Environmental education officer',
                'Conservator, furniture', 'Analytical chemist',
                'Broadcast engineer', 'Media planner', 'Lawyer',
                'Producer, television/film/video',
                'Armed forces technical officer', 'Engineer, site',
                'Contracting civil engineer', 'Veterinary surgeon',
                'Sales promotion account executive', 'Broadcast journalist',
                'Dancer', 'Forest/woodland manager', 'Personnel officer',
                'Industrial buyer', 'Accountant, chartered',
                'Air traffic controller', 'Careers adviser', 'Information officer',
                'Ship broker', 'Legal secretary', 'Homeopath', 'Solicitor',
                'Warehouse manager'
            )

# Load pre-trained model
def load_model(model_path):
    return joblib.load(model_path)

saved_columns = joblib.load('model_columns.pkl')

def preprocess_input(input_df):
    #1. Encode categorical variable
    category_df = pd.get_dummies(input_df['category'], prefix='category').astype(int)
    input_df.drop('category', inplace=True, axis=1)
    
    state_df = pd.get_dummies(input_df['state'], prefix='state').astype(int)
    input_df.drop('state', inplace=True, axis=1)
    
    job_df = pd.get_dummies(input_df['job'], prefix='job').astype(int)
    input_df.drop('job', inplace=True, axis=1)
    
    input_df['gender'].replace({'M': 0, 'F': 1}, inplace=True)
    
    scaler = MinMaxScaler()
    input_df['amount'] = scaler.fit_transform(input_df[['amount']])
    input_df['city_pop'] = scaler.fit_transform(input_df[['city_pop']])
    
    input_df = pd.concat([input_df, category_df, state_df, job_df], axis=1)
    
    input_df = input_df.reindex(columns=saved_columns, fill_value=0)
    return input_df.values

def main():
    # Title and decription
    st.title("Credit Card Fraud Detection App (CFraud APP)")
    st.write("Enter transaction details to predict wheter it's fraudulent.")
    
    # Sidebar for model selection
    st.sidebar.title("Select Model")
    model_choice = st.sidebar.radio(
        "Choose a model to predict:",
        ("Model 1: Logistic Regression", "Model 2: Random Forest", "Model: Artificial Neural Network")
    )
    
    # Input fields for user to enter transaction details
    st.subheader("Enter Transaction Details:")
    
    st.sidebar.markdown("""
        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
        """
    )
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
    
    st.write("Input Data Preview")
    st.write(input_data)
    
    # Load the selected model
    if model_choice == "Model 1: Logistic Regression":
        model = load_model("../models/lg_model1.pt")
    elif model_choice == "Model 2: Random Forest":
        model = load_model("../models/rf_model1.pt")
    elif model_choice == "Model: Artificial Neural Network":
        # from cfraud import CFraud  
        model = CFraud(layers_sz=[300, 150], in_sz=562, out_sz=1)
        model = torch.load('../models/ann_model_update_10.pt')
        model.eval()

        
    # Predict button
    if st.button("Predict"):
        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)
        
        if model_choice == "Model: Artificial Neural Network":
            # Convert to tensor for PyTorch model
            preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float32)
            preprocessed_data = preprocessed_data.unsqueeze(0)  # Ensure batch dimension for a single input
            
            # Perform prediction with ANN
            with torch.no_grad():
                outputs = model(preprocessed_data)  # Get raw outputs
                fraud_prob = torch.sigmoid(outputs).item()  # Sigmoid for binary probability
                prediction = 1 if fraud_prob >= 0.5 else 0  # Threshold for binary classification
        else:
            # For other models (e.g., logistic regression, random forest)
            prediction = model.predict(preprocessed_data)
            prediction_prob = model.predict_proba(preprocessed_data)
            fraud_prob = prediction_prob[0][1]  # Probability of "Fraud"

        # Compute complementary probability
        not_fraud_prob = 1 - fraud_prob

        # Display the result
        st.write("Prediction:", "Fraud" if prediction == 1 else "Not Fraud")
        st.write(f"Prediction Probability (Not Fraud): {not_fraud_prob:.2f}")
        st.write(f"Prediction Probability (Fraud): {fraud_prob:.2f}")


# Run the app
if __name__ == "__main__":
    main()