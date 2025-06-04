\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{Auto-Prompting Framework}
\date{}

\begin{document}
\maketitle

\section*{Auto-Prompting Framework}
\textbf{Auto-Prompting Framework} is a universal tool for automated prompt generation, optimization, and evaluation for large language models (LLMs).

\section{About the Project}
This project implements a framework for automatically developing prompts aimed at solving text classification and generation tasks. Main features include:
\begin{itemize}
    \item \textbf{Prompt Optimization Methods}:
    \begin{itemize}
        \item PromptV1 and PromptV2 (hard token prompts).
        \item Mixture Soft Prompting (blended soft tokens).
        \item TextGrad (gradient-based optimization).
        \item Consprompt (contrastive optimization).
        \item Prewrite RL (reinforcement learning).
        \item APE (automated evolutionary search).
    \end{itemize}
    \item \textbf{Experiment Support} via Hydra + YAML configurations.
    \item \textbf{Integration} with \texttt{transformers}, \texttt{datasets}, \texttt{evaluate}, and \texttt{openai}.
    \item \textbf{Unit Tests} based on \texttt{pytest} for each method implementation.
\end{itemize}

\section{Examples of Using a Custom Method}
The implementation of our custom method for Falcon and RoBERTa-base on AG News, SST-2, and TREC can be found in the notebooks:
\begin{itemize}
    \item \texttt{src/methods/ptcprl\_method\_roberta.ipynb}
    \item \texttt{src/methods/ptcprl\_method\_falcon.ipynb}
\end{itemize}

\section{Dependencies}
\begin{verbatim}
> pytest
> torch
> transformers
> datasets
> evaluate
> hydra-core
> omegaconf
> tqdm
> openai
\end{verbatim}

\section{Project Structure}
\begin{verbatim}
.
├── configs/               # YAML configuration files for experiments
├── src/                   # Source code of the framework
│   ├── run_experiment.py  # Hydra entry point
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── callbacks/         # Metric tracking callbacks per epoch
│   └── methods/           # Implementations of prompt optimization methods
├── utils/                 # Helper functions
├── tests/                 # Tests (pytest)
├── run_all_experiments.ps1# Script to run all configurations (Windows)
├── pytest.ini             # Pytest configuration
└── README.md              # This file
\end{verbatim}

\section{Usage}
\subsection{Configuration}
All \texttt{.yaml} files in the \texttt{configs} folder describe the combination:
\begin{verbatim}
<dataset>_<method>_<model>.yaml
\end{verbatim}
For example: \texttt{ag\_news\_prompt\_v1\_roberta\_base.yaml}.

\subsection{Running an Experiment}
\begin{verbatim}
python src/run_experiment.py --config-name ag_news_prompt_v1_roberta_base
\end{verbatim}
To partially override parameters:
\begin{verbatim}
python src/run_experiment.py --config-name ag_news_prompt_v1_roberta_base \
    method_cfg.num_candidates=10 training.max_epochs=3
\end{verbatim}

\subsection{Results}
\begin{itemize}
    \item Training logs are saved to \texttt{outputs/exp\_name/YYYY-MM-DD\_HH-MM-SS/}.
    \item Epoch metrics are stored in \texttt{epoch\_metrics.jsonl}.
\end{itemize}

\section{Testing}
\begin{verbatim}
pytest
\end{verbatim}

\section{Adding a New Method}
\begin{enumerate}
    \item Create a new Runner class in \texttt{src/methods}, following the pattern of existing ones.
    \item Add an entry to the \texttt{\_METHODS} dictionary in \texttt{src/run\_experiment.py}.
    \item Write a YAML config file in \texttt{configs}.
    \item Add tests under \texttt{tests/}.
\end{enumerate}

\end{document}
