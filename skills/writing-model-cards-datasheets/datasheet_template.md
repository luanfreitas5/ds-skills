# Datasheet — {{nome_do_dataset}}

## 1. Motivação
- Por que foi criado? Para qual tarefa? {{...}}
- Quem criou e quem financiou? {{...}}

## 2. Composição
- O que cada instância representa? {{cliente, transação, internação...}}
- Número de instâncias e período coberto: {{...}}
- Campos, tipos e contrato: `src/schemas/{{schema}}.py`
- Rótulo: definição exata e janela temporal: {{...}}
- Dados ausentes (taxa por coluna): {{...}}
- Campos sensíveis / PII e tratamento: {{...}}
- É amostra de uma população maior? Como amostrado? {{...}}

## 3. Processo de coleta
- Fonte(s) e sistema(s) de origem: {{...}}
- Período e frequência de coleta: {{...}}
- Consentimento / aviso aos titulares: {{...}}

## 4. Pré-processamento
- Limpeza, filtragem e linhas removidas por regra: {{...}}
- Anonimização / pseudonimização: {{método, onde fica o segredo}}
- Hash SHA-256 dos dados brutos: `{{sha256}}`

## 5. Usos
- Usos pretendidos: {{...}}
- Usos desaconselhados: {{...}}
- Riscos de viés conhecidos: {{...}}

## 6. Distribuição
- Licença e restrições: {{...}}
- Onde está armazenado (DVC remote, bucket): {{...}}

## 7. Manutenção
- Responsável: {{...}}
- Frequência de atualização: {{...}}
- Como atender pedidos de exclusão de titulares: {{...}}

## 8. LGPD
- Base legal (art. 7 / art. 11): {{...}}
- Prazo de retenção: {{...}}
- Relatório de impacto (RIPD) necessário? {{sim/não, link}}
