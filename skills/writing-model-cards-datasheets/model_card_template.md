# Model Card — {{nome_do_modelo}} v{{versao}}

**Status:** DRAFT | APPROVED FOR {{uso específico}} | DEPRECATED
**Bloqueios pendentes:** {{lista de seções TODO e qual uso cada uma bloqueia}}

## 1. Detalhes do modelo
- **Responsável / contato:** {{nome, e-mail, canal}}
- **Data:** {{AAAA-MM-DD}}
- **Algoritmo:** {{ex.: LightGBM 4.5, pipeline sklearn completo}}
- **Rastreabilidade:** Git SHA `{{sha}}` · MLflow run `{{run_id}}` · hash dos dados `{{sha256}}`
- **Licença:** {{licença}}

## 2. Uso pretendido
- **Uso primário:** {{decisão apoiada pelo modelo}}
- **Usuários pretendidos:** {{quem consome o score}}
- **Humano no circuito:** {{sim/não, como}}

## 3. Usos fora do escopo
- {{uso proibido 1 — ex.: negar atendimento automaticamente}}
- {{uso não validado — ex.: hospitais fora da rede de treino}}

## 4. Fatores
- **Grupos relevantes:** {{idade, sexo, região, unidade...}}
- **Ambientes:** {{canais, períodos, sistemas de origem}}

## 5. Métricas
- **Métrica principal:** {{métrica}} — justificativa: {{custo de FP vs FN}}
- **Limiar de decisão:** {{valor}} — escolhido em {{validação}}, por {{critério}}
- **Calibração:** Brier {{valor}}; curva em `reports/figures/{{arquivo}}`

## 6. Dados de avaliação
{{fonte, período, tamanho, prevalência, relação temporal com o treino}}

## 7. Dados de treino
{{resumo; link para datasheet em reports/datasheets/}}

## 8. Análise quantitativa
| Recorte | n | Métrica ± IC 95% | Baseline |
|---|---|---|---|
| Geral | | | |
| {{grupo}} | | | |

## 9. Considerações éticas e fairness
- Atributos sensíveis auditados: {{lista}}
- Disparidades encontradas: {{métrica, valor, limiar}}
- Mitigação aplicada: {{nenhuma / método}}
- Base legal LGPD: {{base}}

## 10. Limitações, riscos e monitoramento
- **Limitações conhecidas:** {{lista}}
- **Mudanças de distribuição previstas:** {{ex.: sazonalidade, pandemia}}
- **Monitoramento:** {{drift de dados/predição, métrica com rótulos, frequência, alerta}}
- **Critério de retreino / descontinuação:** {{critério}}
