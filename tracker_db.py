import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
import sqlite3

# Define the path to the SQLite database
db_path = 'student_MH_data.db'

# Initialize the database and create table if it doesn't exist
def initialize_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_log (
            id INTEGER PRIMARY KEY, 
            log_date DATE NOT NULL,
            mood_log INTEGER,
            sleep INTEGER,
            uni_work INTEGER,
            other_work INTEGER,
            coping INTEGER,
            confidence INTEGER,
            struggles TEXT,
            contact_hrs INTEGER,
            journal_entry TEXT,
            entry_score INTEGER
        );
    """)
    conn.commit()
    conn.close()

# Insert data into the user_log table
def create_table(answer_list1, answer_list2):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')

    cursor.execute("""
        INSERT INTO user_log (log_date, mood_log, sleep, uni_work, other_work, coping, confidence, struggles, contact_hrs, journal_entry, entry_score) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date_string, answer_list1[0], answer_list1[1], answer_list2[0], answer_list2[1], answer_list2[2], answer_list2[3],
        answer_list1[2], answer_list1[3], answer_list1[4], answer_list1[5]
    ))

    conn.commit()
    conn.close()

# Read the table and return data as a pandas DataFrame
def read_table(columns=None):
    if columns is None:
        columns = '*'
    query = f'SELECT {columns} FROM user_log ORDER BY log_date DESC'
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df

# Visualize data with Streamlit and Plotly
def visualise_data():
    df = read_table()

    # Convert columns to appropriate data types
    numeric_columns = ['mood_log', 'sleep', 'uni_work', 'other_work', 'coping', 'confidence', 'contact_hrs',
                       'entry_score']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, set invalid values to NaN

    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['log_date', 'mood_log', 'contact_hrs', 'entry_score'])

    # Ensure 'log_date' is a datetime object
    df['log_date'] = pd.to_datetime(df['log_date'])

    # Plot 1: Mood, Sleep, and Other Activities
    fig1 = px.line(df, x='log_date', y=['sleep', 'coping', 'uni_work', 'other_work'],
                   line_shape='spline', title='Mental Health Activity Over Time')
    fig1.update_layout(xaxis_tickformat='%Y-%m-%d')
    st.plotly_chart(fig1)

    # Plot 2: Entry Score
    fig2 = px.line(df, x='log_date', y='entry_score',
                   line_shape='spline', title='Entry Score Over Time')
    fig2.update_layout(xaxis_tickformat='%Y-%m-%d')
    st.plotly_chart(fig2)

    # Plot 3: Mood and Contact Hours
    fig3 = px.line(df, x='log_date', y=['mood_log', 'contact_hrs'],
                   line_shape='spline', title='Mood and Contact Hours Over Time')
    fig3.update_layout(xaxis_tickformat='%Y-%m-%d')
    st.plotly_chart(fig3)


# Function to get user input via Streamlit widgets
def get_user_input():
    answer_list1 = [
        st.slider("Mood (1-10)", 1, 10, 5),  # Mood log
        st.number_input("Sleep Hours", min_value=0, value=8),  # Sleep hours
        st.text_area("Struggles", "None"),  # Struggles
        st.number_input("Contact Hours", min_value=0, value=0),  # Contact hours
        st.text_area("Journal Entry", "No entry"),  # Journal entry
        st.slider("Entry Score (0-100)", 0, 100, 50)  # Entry score
    ]

    answer_list2 = [
        st.slider("University Work (1-10)", 1, 10, 5),  # University work
        st.slider("Other Work (1-10)", 1, 10, 5),  # Other work
        st.slider("Coping (1-10)", 1, 10, 5),  # Coping
        st.slider("Confidence (1-10)", 1, 10, 5)  # Confidence
    ]

    return answer_list1, answer_list2




# Streamlit app layout
def main():
    st.title("Student Mental Health Tracker")

    # Initialize database
    initialize_db()

    # Collect user input within a form
    with st.form("User Input Form"):
        answer_list1, answer_list2 = get_user_input()
        submit = st.form_submit_button("Submit Data")

        if submit:
            try:
                # Validate that all inputs are populated
                if len(answer_list1) < 6 or len(answer_list2) < 4:
                    st.error("Please fill out all fields before submitting.")
                else:
                    create_table(answer_list1, answer_list2)
                    st.success("Data submitted successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Visualize the data
    if st.button("Visualize Data"):
        visualise_data()



if __name__ == "__main__":
    main()
