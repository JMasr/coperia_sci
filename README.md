# COPERIA: AI Models for Voice Signal Analysis in PASC Patients

## About

This repository provides an implementation for the data pipelines and AI models used in the COPERIA project.

The [COPERIA project](https://coperia.es/) aims to develop and clinically validate a comprehensive multidisciplinary
platform that utilizes artificial intelligence for the diagnosis,
empowerment, and clinical management of Post-Acute Sequelae of SARS-CoV-2 (PASC) patients.
The clinical study conducted in the context of the COPERIA project received ethical approval from the Clinical Research
Ethics Committee of Galicia,
and all procedures were conducted in compliance with the ethical principles outlined in the Declaration of Helsinki.
Informed consent was obtained from all participants prior to their involvement in the study.
The study was registered in the US Clinical Trials Registry under the code [NCT05629793].

The project is developed by the [Multimedia Technologies Group](https://gtm.uvigo.es/en/) at the **atlanTTic Research
Center, Universidade de Vigo**, in collaboration with the "Persistent COVID Unit of the Ourense Hospital" and primary
care centers in the health area.

## Features

* Data pipelines for processing voice and metadata data from the COPERIA project.
* AI models for voice signal analysis in PASC patients.
* Data visualization tools for exploring the data.
* Data analysis tools for extracting insights from the data.

## Getting Started

**Recommended Python version: 3.9**

1. Clone the repository:

```bash
git clone https://github.com/JMasr/corilga_api.git
```

2. Navigate to the project directory:

```bash
cd corilga_api
```

3. Import the envairoment using conda:

```bash
conda env create -f env.yml
```

4. Install requirements:

```bash
pip install -r requirements.txt
```

5. Use **exp_config.json** to set the paths to the data and models.

## Acknowledgements

*. This work was supported by the CONECTA COVID programme, co-financed by the European Regional Development Fund (ERDF)
within the Galicia ERDF operational programme 2014-2020 as part of the EU’s response to the COVID19 pandemic, and
Axencia Galega de Innovacion (GAIN).

*. This work has received financial support from the Xunta de Galicia (Centro singular de investigación de Galicia
accreditation 2019-2022).

*. This research has been funded by the Galician Regional Government under project ED431B 2021/24“GPC".

*. Thanks to the “Unidad de COVID Persistente del Hospital de Ourense” and the patients involved in the study.
