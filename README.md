# Spatio-Temporal Deep Networks with Feature Disentangling for Advancing Earthquake Monitoring

## Abstract
Earthquake monitoring is essential for assessing seismic hazards and involves interconnected tasks such as phase picking and location estimation. Existing single-parameter estimation methods suffer from error accumulation caused by task interdependencies and typically rely on empirical values. Multi-parameter estimation methods ofen depend on data from multiple stations, posing challenges in modeling and revealing the inter-station relationships. To address these challenges, this study proposes a novel neural network, SINE, designed to simultaneously estimate key parameters in earthquake monitoring, including p-phase arrival time, location, and magnitude. SINE develops a multi-task framework that incorporates Graph Neural Networks (GNNs) and Bidirectional Long Short-Term Memory networks (BI-LSTM) to extract spatio-temporal features, effectively mitigating error accumulation across the tasks. Unlike previous GNN-based models, SINE incorporates a feature disentanglement structure to automatically identify multiple potential relationships between seismic stations. Additionally, the CNN-based parsing unit is employed to regress multiple seismic parameters simultaneously. Evaluation on datasets from Southern California and Italy shows that SINE outperforms existing DL models and traditional seismological methods. Furthermore, SINE effectively reduces inter-task dependencies, enhancing robustness in earthquake monitoring.
## Features
- **Training and Testing:**
  - `train.py`: Script for training the SINE model.
  - `test.py`: Script for testing the trained model.
- **Configuration:**
  - All training configurations are stored in the `Utils` folder:
    - `utils.py`: Contains utility functions for data processing and model management.
    - `config.ini`: Stores the parameters used to process continuous data.
- **Real-time Data Processing:**
  - `Continuous_data.py`: Supports real-time data sliding window detection and computation. Currently, it only reads local files, with future support planned for Redis and Kafka integration.
- **Dependency Management:**
  - `requirements.txt`: Lists all necessary packages for running the project.
- **Dataset Preparation:**
  - Users can create custom datasets tailored to their data format.
  - Sample datasets are provided in the `ExampleData` folder for reference.

## Installation
Ensure you have Python installed, then clone the repository and install dependencies:

```bash
# Clone the repository
git clone <repository_url>
cd SINE

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run:

```bash
python train.py
```

Ensure your training parameters are set correctly.

### Testing the Model
To test the trained model, run:

```bash
python test.py
```

### Real-time Data Processing
For real-time sliding window detection:

```bash
python Continuous_data.py
```

(Currently supports local file processing; future versions will include Redis and Kafka support.)

## Dataset
Create your dataset following the format used in `ExampleData`. Update `config.ini` with your dataset path.

## Roadmap
- [ ] Enhance model performance with additional training strategies
- [ ] Integrate Redis for real-time data streaming
- [ ] Add Kafka support



