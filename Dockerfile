FROM continuumio/miniconda3:latest

WORKDIR /app

# Install g++ (needed to build IsoSpecPy C++ extension)
RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*

# Install conda dependencies (including RDKit from conda-forge)
COPY environment_hf.yml .
RUN conda env create -f environment_hf.yml && conda clean -afy

# Activate conda env for all subsequent commands
SHELL ["conda", "run", "-n", "dna_mass_spec", "/bin/bash", "-c"]

# Copy application code
COPY dna_silver_webapp.py .
COPY core/ core/
COPY lib/ lib/
COPY templates/index.html templates/
COPY sample_data/ sample_data/

# HF Spaces expects port 7860
ENV PORT=7860
ENV SECRET_KEY=auto
ENV FLASK_ENV=production

EXPOSE 7860

CMD ["conda", "run", "--no-capture-output", "-n", "dna_mass_spec", "python", "dna_silver_webapp.py"]
