# ATV Maintenance and Analytics System

A comprehensive system for military ATV maintenance management, defect analysis, and predictive maintenance using AI.

## Features

- 🤖 AI-Powered Maintenance Chatbot
- 📊 Real-time Analytics Dashboard
- 🔍 Defect Pattern Analysis
- 📈 Spare Parts Forecasting
- 📝 Maintenance History Tracking

## Supported ATV Models

- Ranger-1000
- MRZR
- RZR

## Tech Stack

- Python 3.11+
- Streamlit
- Google Gemini AI
- ChromaDB
- Plotly
- Pandas

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/atv-maintenance-system.git
   cd atv-maintenance-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Running Locally

```bash
streamlit run chatbot_interface.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add your environment variables in Streamlit Cloud settings
5. Deploy!

## Project Structure

```
atv_analytics/
├── chatbot_interface.py    # Main Streamlit interface
├── agent.py               # AI agent implementation
├── knowledge_base.py      # ChromaDB integration
├── memory_manager.py      # Conversation management
├── requirements.txt       # Project dependencies
└── README.md             # Documentation
```

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key

## Data Storage

- Maintenance records are stored in ChromaDB
- Conversation history is persisted locally
- Analytics data is generated in real-time

## Security Notes

- Never commit your `.env` file
- Keep your API keys secure
- Regularly update dependencies
- Monitor access logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Support

For support, please open an issue in the GitHub repository or contact the maintenance team. 