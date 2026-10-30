# 🧪 ds-skills — Skills de Ciência de Dados para Claude Code

![License](https://img.shields.io/badge/license-MIT-green) ![Skills](https://img.shields.io/badge/skills-78-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Coleção de skills que levam agentes de IA ao padrão de um cientista de dados sênior: avaliação estatisticamente correta, dados validados, resultados reprodutíveis e IA responsável (LGPD, fairness, documentação).

Cada skill foi escrita usando a skill [superpowers:writing-skills](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md) com modelo `Claude Opus 5.5 High`, seguindo TDD para documentação: um cenário de pressão foi executado **sem** a skill (baseline), as falhas observadas guiaram o texto, e o mesmo cenário foi repetido **com** a skill para verificar a mudança de comportamento. Todos os trechos de código foram executados contra dados sintéticos.

![ds-skills](banner/skills.png)

## 📦 Skills

| Grupo | Skill | Quando é acionada |
|---|---|---|
| **🏗️ Engenharia do projeto** | [`scaffolding-ds-projects`](skills/scaffolding-ds-projects/SKILL.md) | Novo projeto, `pyproject.toml`, pre-commit, CI |
|  | [`building-ds-github-actions`](skills/building-ds-github-actions/SKILL.md) | Criar ou corrigir `ci.yml`, `tests.yml`, `docs.yml` |
|  | [`testing-ml-code`](skills/testing-ml-code/SKILL.md) | Testes de transformações e modelos treinados |
|  | [`refactoring-notebooks-to-src`](skills/refactoring-notebooks-to-src/SKILL.md) | Lógica de notebook vira pipeline; "limpar" um notebook |
|  | [`configuring-typed-settings`](skills/configuring-typed-settings/SKILL.md) | Configs YAML, argparse, `.env`; hiperparâmetro ou chave nova |
|  | [`ensuring-reproducibility`](skills/ensuring-reproducibility/SKILL.md) | Scripts de treino para artigo, revisão ou deploy |
|  | [`setting-up-project-logging`](skills/setting-up-project-logging/SKILL.md) | Configurar logging; `print()` para acompanhar execução; logs duplicados ou barra rich quebrada |
|  | [`designing-exception-hierarchies`](skills/designing-exception-hierarchies/SKILL.md) | Tratamento de erros em pipelines; `except Exception` genérico; `errors="coerce"` sem limite |
|  | [`type-checking-with-basedpyright`](skills/type-checking-with-basedpyright/SKILL.md) | Configurar checagem de tipos; erros de tipo com pandas/polars/numpy; vontade de usar `# type: ignore` ou `Any` |
|  | [`fixing-python-warnings`](skills/fixing-python-warnings/SKILL.md) | `FutureWarning`, `DeprecationWarning`, `RuntimeWarning` ou `ConvergenceWarning` no log/pytest; pedido de "sumir com os warnings"; vontade de usar `filterwarnings("ignore")` ou `np.errstate` |
|  | [`auditing-python-dependencies`](skills/auditing-python-dependencies/SKILL.md) | Adicionar dependência; CVE no `pip-audit`; revisão periódica de dependências |
| **📝 Documentação e colaboração** | [`writing-numpy-docstrings`](skills/writing-numpy-docstrings/SKILL.md) | Documentar funções públicas; `interrogate` falhando no CI; Examples que não rodam |
|  | [`writing-conventional-commits`](skills/writing-conventional-commits/SKILL.md) | Commitar, preparar release ou gerar CHANGELOG; commit "update" misturando dado, modelo e refatoração |
|  | [`writing-ds-pull-requests`](skills/writing-ds-pull-requests/SKILL.md) | Abrir PR que muda dados, features ou modelo; "métricas depois" |
|  | [`reviewing-ds-code`](skills/reviewing-ds-code/SKILL.md) | Revisar PR ou diff de código de ciência de dados |
|  | [`writing-portfolio-readmes`](skills/writing-portfolio-readmes/SKILL.md) | README de portfólio ou de artigo; pedido de "listar as tecnologias", badges, instalação |
| **🔎 Dados e análise** | [`validating-data-contracts`](skills/validating-data-contracts/SKILL.md) | Funções que leem/limpam/gravam dados entre estágios |
|  | [`running-structured-eda`](skills/running-structured-eda/SKILL.md) | Primeira exploração de dataset novo (`01_eda.ipynb`); "EDA padrão" de `describe()` e histogramas |
|  | [`writing-analytical-sql`](skills/writing-analytical-sql/SKILL.md) | SQL de análise ou extração de features; JOIN 1:N, `NOT IN`, janelas |
|  | [`optimizing-sql-queries`](skills/optimizing-sql-queries/SKILL.md) | Consulta lenta ou cara no warehouse; antes de criar índice, partição ou clustering |
|  | [`querying-with-duckdb`](skills/querying-with-duckdb/SKILL.md) | Análise local sobre Parquet/CSV grandes; pandas não cabe na memória; DuckDB × polars lazy |
|  | [`writing-idiomatic-polars`](skills/writing-idiomatic-polars/SKILL.md) | Código polars, migração de pandas, dados maiores que a RAM |
| **🛢️ Engenharia de dados** | [`ingesting-external-data`](skills/ingesting-external-data/SKILL.md) | Baixar dados de APIs, bancos ou sites para `data/bronze` (Medallion); jobs de coleta agendados |
|  | [`building-point-in-time-features`](skills/building-point-in-time-features/SKILL.md) | Features de tabelas históricas para rótulos com data |
|  | [`versioning-data-with-dvc`](skills/versioning-data-with-dvc/SKILL.md) | Versionar dados/modelos; alguém quer commitar `.parquet` ou `.joblib` |
|  | [`profiling-data-pipelines`](skills/profiling-data-pipelines/SKILL.md) | Pipeline lento ou estourando memória, antes de otimizar; "reescrever em polars" ou paralelizar sem medir |
|  | [`writing-pyspark`](skills/writing-pyspark/SKILL.md) | Job PySpark lento ou com skew; dados maiores que uma máquina; UDF Python linha a linha; `collect()` ou `toPandas()` em tabela grande; "mais executores" sem medir |
| **⚙️ Orquestração e automação** | [`orchestrating-pipelines`](skills/orchestrating-pipelines/SKILL.md) | Cron chamando um script; reexecução duplica linhas ou sobrescreve `data/raw`; falha silenciosa; backfill; Prefect, Dagster ou Airflow |
|  | [`building-n8n-workflows`](skills/building-n8n-workflows/SKILL.md) | Workflow N8N com LLM, webhooks e agendamentos; chaves coladas nos nós; reexecução que reenvia e-mails; ramo de erro "desnecessário" |
| **📊 Visualização** | [`plotting-publication-figures`](skills/plotting-publication-figures/SKILL.md) | Gráficos para relatório, artigo ou README |
|  | [`building-streamlit-apps`](skills/building-streamlit-apps/SKILL.md) | Criar ou alterar dashboard Streamlit |
| **📐 Estatística e inferência causal** | [`designing-ab-tests`](skills/designing-ab-tests/SKILL.md) | Planejar ou analisar teste A/B; p-valor olhado todo dia; grupos desbalanceados |
|  | [`estimating-treatment-effects`](skills/estimating-treatment-effects/SKILL.md) | Efeito de campanha ou benefício sem experimento aleatorizado; "a correlação mostra que funcionou"; regressão com todas as variáveis, inclusive pós-tratamento; propensity score, DoWhy, econml, diff-in-diff |
| **🎯 Avaliação e rigor de modelagem** | [`establishing-baselines`](skills/establishing-baselines/SKILL.md) | Início de modelagem, algoritmo "já escolhido", métrica sem referência |
|  | [`evaluating-models-rigorously`](skills/evaluating-models-rigorously/SKILL.md) | Reportar métricas, comparar modelos, declarar "melhor modelo" |
|  | [`detecting-data-leakage`](skills/detecting-data-leakage/SKILL.md) | Métrica alta demais, snapshot posterior ao rótulo, pré-processamento antes do split |
|  | [`checking-probability-calibration`](skills/checking-probability-calibration/SKILL.md) | Probabilidades usadas para escolher limiar ou precificar risco; "score 0,30 = 30% de chance"; curva de calibração e Brier; Platt × isotonic; predição conformal |
|  | [`explaining-models-shap`](skills/explaining-models-shap/SKILL.md) | Explicar previsões ou importância de features; features correlacionadas |
| **🌲 ML clássico (tabular)** | [`training-gradient-boosting`](skills/training-gradient-boosting/SKILL.md) | LightGBM, XGBoost ou CatBoost em dados tabulares; early stopping no mesmo conjunto que reporta a métrica; target encoding antes do split; one-hot em categórica de alta cardinalidade; refit com a melhor iteração |
|  | [`handling-class-imbalance`](skills/handling-class-imbalance/SKILL.md) | Classe positiva rara (fraude, churn, doença); SMOTE proposto; acurácia ~0,99 em alvo raro; escolha de limiar |
|  | [`tuning-hyperparameters-optuna`](skills/tuning-hyperparameters-optuna/SKILL.md) | Ajuste de hiperparâmetros; melhor score da busca reportado como desempenho |
| **📈 Séries temporais** | [`backtesting-time-series`](skills/backtesting-time-series/SKILL.md) | Avaliar previsão temporal; KFold com shuffle em série; horizonte ou gap diferentes da produção |
|  | [`forecasting-time-series`](skills/forecasting-time-series/SKILL.md) | Previsão com statsforecast, sktime ou Prophet; baseline sazonal; várias séries; intervalos de previsão; métrica em série com zeros |
| **💬 NLP** | [`building-nlp-classifiers-ptbr`](skills/building-nlp-classifiers-ptbr/SKILL.md) | Classificar texto de redes sociais em pt-BR; rótulos de dois ou mais anotadores; gírias, emojis e erros de grafia; "limpar" removendo emojis e stopwords; escolher entre TF-IDF e Transformer |
|  | [`fine-tuning-transformers-hf`](skills/fine-tuning-transformers-hf/SKILL.md) | Fine-tuning de BERT/RoBERTa/LLM com transformers, peft (LoRA, QLoRA) ou trl; "treinar mais épocas"; reportar o F1 do melhor checkpoint; teste usado como `eval_dataset` |
| **🤖 IA generativa e LLMs** | [`building-rag-pipelines`](skills/building-rag-pipelines/SKILL.md) | Construir RAG ou Q&A sobre documentos; escolher chunk, embedding e top-k; avaliar só a resposta final |
|  | [`building-llm-agents-langgraph`](skills/building-llm-agents-langgraph/SKILL.md) | Agente com tools, estado e loops (LangGraph); "agente autônomo que resolve tudo"; sem limite de passos, custo ou tempo; testar trajetória do agente |
|  | [`treating-external-content-as-data`](skills/treating-external-content-as-data/SKILL.md) | RAG ou agente que lê documentos, e-mails ou páginas não confiáveis; tools com efeito colateral; saída do LLM executada; "prompt de sistema mais firme" como defesa contra injeção |
|  | [`evaluating-llm-outputs`](skills/evaluating-llm-outputs/SKILL.md) | Medir qualidade de prompt, agente ou pipeline com LLM; LLM como juiz |
|  | [`labeling-with-llms`](skills/labeling-with-llms/SKILL.md) | Usar LLM para rotular dados ou gerar dados sintéticos; texto que pode conter dados pessoais |
|  | [`versioning-prompts`](skills/versioning-prompts/SKILL.md) | Prompt em produção muda com frequência; ajuste "de uma linha"; saída sem versão do prompt |
|  | [`tracing-llm-apps`](skills/tracing-llm-apps/SKILL.md) | App com LLM em produção sem visibilidade de custo e latência; depurar resposta ruim; comparar versões de prompt |
|  | [`serving-llms-vllm-ollama`](skills/serving-llms-vllm-ollama/SKILL.md) | Servir LLM local ou em GPU; quantização; Ollama em produção; estouro de VRAM; throughput × latência |
| **🛒 Sistemas de recomendação** | [`splitting-recsys-data`](skills/splitting-recsys-data/SKILL.md) | Split de interações para recomendador; split temporal por usuário ou leave-last-out; cold start de usuário e de item; filtro de mínimo de interações; embeddings treinados com interações futuras |
|  | [`building-recsys-baselines`](skills/building-recsys-baselines/SKILL.md) | Início de projeto de recomendação; "já vou usar deep learning"; baselines de popularidade, co-ocorrência, itemKNN e ALS (implicit, LightFM, Surprise); comparar só com a versão anterior do modelo |
|  | [`handling-implicit-feedback`](skills/handling-implicit-feedback/SKILL.md) | Recomendador com cliques, views, compras ou plays sem nota; eventos mapeados para notas falsas; itens sem interação preenchidos com 0; regressão com MSE/RMSE em cliques; negativos amostrados; pesos de confiança do ALS |
|  | [`building-two-tower-retrieval`](skills/building-two-tower-retrieval/SKILL.md) | Modelo de ranking pesado sobre milhões de itens; embeddings de usuário e item para recuperação de candidatos; FAISS ou Qdrant, flat × HNSW/IVF; recall@K da etapa de retrieval; embeddings de item nunca atualizados |
|  | [`evaluating-recommenders-offline`](skills/evaluating-recommenders-offline/SKILL.md) | Reportar NDCG@k, MAP ou recall@k; comparar recomendadores; split aleatório em dados com timestamp; sem baseline de popularidade; negativos amostrados (RecBole, implicit, recommenders) |
|  | [`auditing-recsys-bias`](skills/auditing-recsys-bias/SKILL.md) | Viés de popularidade; loop de feedback; diversidade, novidade e cobertura; exposição de itens ou grupos; "o NDCG subiu, pode publicar" |
|  | [`testing-recsys-online`](skills/testing-recsys-online/SKILL.md) | Validar em produção um ganho offline de NDCG/recall; "lançar para 100%"; A/B randomizado por sessão; CTR da lista inteira ignorando posição; interleaving |
| **👁️ Visão computacional** | [`splitting-image-datasets`](skills/splitting-image-datasets/SKILL.md) | Vários frames ou recortes do mesmo paciente, objeto ou cena; imagens duplicadas ou quase idênticas; GroupKFold; acurácia suspeitosamente alta |
|  | [`labeling-images-with-quality-control`](skills/labeling-images-with-quality-control/SKILL.md) | Anotação manual (Label Studio) ou por modelo; um só anotador; sem guia de rótulos nem amostra dupla; concordância entre anotadores; ruído de rótulo (cleanlab) |
|  | [`augmenting-images-safely`](skills/augmenting-images-safely/SKILL.md) | Augmentation (Albumentations, torchvision v2); flip em dígitos, texto ou lateralidade; mesma transformação em treino e validação; MixUp/CutMix; crop que remove o objeto |
|  | [`fine-tuning-vision-models`](skills/fine-tuning-vision-models/SKILL.md) | Fine-tuning ou transfer learning com backbone de visão pré-treinado (torchvision, timm) em dataset pequeno; CNN do zero com poucas imagens; congelar e descongelar camadas; taxa de aprendizado única; sem baseline de extração de features |
|  | [`evaluating-detection-segmentation`](skills/evaluating-detection-segmentation/SKILL.md) | mAP, IoU, Dice; limiar de confiança; classes raras e objetos pequenos; reportar só acurácia em detecção ou segmentação (ultralytics, torchmetrics, pycocotools) |
|  | [`explaining-vision-models`](skills/explaining-vision-models/SKILL.md) | Grad-CAM, saliency ou atenção mostrados como prova de que o modelo aprendeu certo; mapa acende em canto, marca ou texto; AUC alto em imagem médica indo para piloto; legendas de heatmap (pytorch-grad-cam, captum) |
|  | [`serving-vision-models`](skills/serving-vision-models/SKILL.md) | API de classificador ou detector (FastAPI, ONNX Runtime, Triton); resize cv2 × PIL; upload sem limite; modelo carregado por requisição; latência e batch |
|  | [`building-multimodal-pipelines`](skills/building-multimodal-pipelines/SKILL.md) | CLIP ou SigLIP zero-shot e busca imagem-texto; VLM como rotulador; nomes de classe usados como prompt; "validado" com poucos exemplos escolhidos a dedo |
| **🔁 MLOps** | [`tracking-experiments-mlflow`](skills/tracking-experiments-mlflow/SKILL.md) | Treinar ou comparar modelos em script de experimento |
|  | [`promoting-models-registry`](skills/promoting-models-registry/SKILL.md) | Registrar modelo ou promover para produção |
|  | [`monitoring-model-drift`](skills/monitoring-model-drift/SKILL.md) | Modelos em produção; rótulos com atraso |
| **🚀 Entrega e deploy** | [`serving-models-fastapi`](skills/serving-models-fastapi/SKILL.md) | Expor modelo como API |
|  | [`exporting-models-onnx`](skills/exporting-models-onnx/SKILL.md) | Exportar PyTorch ou scikit-learn para ONNX; reduzir latência ou tamanho; quantização int8; saídas divergentes após a exportação |
|  | [`containerizing-ml-apps`](skills/containerizing-ml-apps/SKILL.md) | Dockerfile ou docker-compose para app, API ou MLflow; imagem grande, root ou modelo dentro da imagem |
|  | [`deploying-to-cloud`](skills/deploying-to-cloud/SKILL.md) | Treino ou deploy em SageMaker, Azure ML ou Vertex AI; permissões IAM; custo de endpoint ocioso; credenciais de nuvem |
| **⚖️ IA responsável e LGPD** | [`auditing-fairness`](skills/auditing-fairness/SKILL.md) | Modelos que afetam pessoas; atributos sensíveis |
|  | [`protecting-pii-lgpd`](skills/protecting-pii-lgpd/SKILL.md) | Dados pessoais em logs, exports, relatórios |
|  | [`writing-model-cards-datasheets`](skills/writing-model-cards-datasheets/SKILL.md) | Documentar modelos e datasets |

## 🚀 Instalação

Como plugin (recomendado):

```text
/plugin marketplace add https://github.com/luanfreitas5/ds-skills.git
/plugin install ds-skills@ds-skills-marketplace
```

Ou como skills pessoais: copie as pastas de `skills/` para `~/.claude/skills/`.

## 🗂️ Estrutura

```
ds-skills/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── skills/
│   └── <skill-name>/
│       ├── SKILL.md          # frontmatter (name, description) + conteúdo
│       └── *_template.*      # templates reutilizáveis, quando aplicável
├── CLAUDE.md                 # padrões de projeto que originaram as skills
├── LICENSE
└── README.md
```

## ✍️ Convenções

- Texto das skills em inglês (melhor descoberta pelo agente); exemplos de código com docstrings NumPy e logs em pt-BR.
- `description` descreve **quando** usar, nunca o passo a passo.
- Referências cruzadas via `REQUIRED SUB-SKILL: <nome>`.

## 📄 Licença

MIT — veja [LICENSE](LICENSE).
