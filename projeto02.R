# Definindo a pasta de trabalho
#setwd("/Users/robsonbonfim/Library/Mobile Documents/com~apple~CloudDocs/Education/dsa/FCD/01-BigDataRAzure/Cap20/Projeto2")
#getwd()

# Install packages
packages <- c('tidymodels', 'readxl', 'corrplot', 'dplyr', 'caret', 'ROCR', 'e1071', 'ggplot2')
install.packages(setdiff(packages, rownames(installed.packages())))

# Importando as bibliotecas necessárias
library(readxl)
library(corrplot)
library(dplyr)
library(caret)
library(ROCR) 
library(e1071) 
library(ggplot2)
library(tidymodels)

set.seed(227)

# Leitura dos dados
df = read_excel("data/Acoustic_Extinguisher_Fire_Dataset.xlsx") 

# Exploração Inicial dos Dados
dim(df)
# 17442 7

str(df)
View(df)

# Procurando valores N/A no dataset
names(which(colSums(is.na(df))>0))

# Tudo certo!!

# Transformar as variaveis String em categoricas
df$FUEL = as.factor(df$FUEL)
df$STATUS = as.factor(df$STATUS)
df$SIZE = as.factor(df$SIZE)

str(df)
summary(df)

# Vou criar um subset com as variasveis numericas para fazer o mapa de correlação
df_num <- select(df,-c('FUEL','STATUS','SIZE'))
df_num

df_corr <- cor(df_num)
corrplot(df_corr)

# Explorando os dados
ggplot(df, aes(x = STATUS)) +
  geom_bar(stat = "count")

ggplot(df, aes(x = STATUS, fill = factor(FUEL))) +
  geom_bar(stat = "count") +
  scale_fill_discrete(
    name = "Type of Fuel",
    labels = c("gasoline", "kerosene", "thinner", "lpg")
  )

ggplot(df, aes(x = STATUS, fill = factor(SIZE))) +
  geom_bar(stat = "count") +
  scale_fill_discrete(
    name = "Type of Size",
    labels = c("1", "2", "3", "4", "5", "6", "7")
  )

size_table <- table(df$SIZE)
size_table

fuel_table <- table(df$FUEL)
fuel_table

distance_table <- table(df$DISTANCE)
distance_table
hist(df$DISTANCE,
     labels = T, 
     main = "Histograma de Distancias", 
     breaks = 19)
# Analisando esta variavel foi possível identificar que foram testados 918 equipamentos, testado em todas distancias a partir de 
# 10cm até 190cm


table(df$DISTANCE, df$STATUS)
# A eficiencia em estinção do Fogo para as menores distancias são consideradas as mais eficientes descartando os outros itens,
# e essa condição se extende até as maiores distancias onde a eficacia é bem reduzida


hist(df$DESIBEL,
     xlab = "Desibeis", 
     main = "Histograma de Decibeis", 
     xlim = range(70:120))

hist(df$AIRFLOW,
     xlab = "Fluxo de Ar", 
     main = "Histograma de Fluxo de Ar", 
     breaks = 15,
     xlim = range(0:20))

hist(df$FREQUENCY,
     xlab = "Frequencia", 
     main = "Histograma de Frequencia", 
     xlim = range(0:80))

# Analisando esta variavel percebi que forma testados 17 equipamentos (anteriormente achei que fosse 918 porém se pegarmos a
# quantidade de frequencias testadas são exatamente iguais são 54 frquencias temos (918/54) 

length(unique(df$FREQUENCY))
table(df$SIZE, df$DISTANCE)
table(df$SIZE, df$FREQUENCY)

# As amostras de LPG foram testadas uma vez para cada distancia/frequencia, a partir dessa analise faz sentido dizer que estas
# duas variaveis tem uma alta correlação 
# 3 tipos de combustivel * 5 tamanhos * 19 distancias * 54 frequecias = 15390 
# 2 tamanhos de LPG * 19 distancias * 54 frequencias = 2052
print((3*5*19*54)+(2*19*54))
dim(df)

# A proporção esta equivalente para o treinamento do modelo
round(prop.table(table(df$STATUS)) * 100, digits = 1) 

# Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Normalizando as variáveis
numeric.vars <- c("DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY")
dataset_scaled <- scale.features(df, numeric.vars)
dataset_scaled

# Aplicando as conversões ao dataset
dataset_final <- dataset_scaled
head(dataset_final)
summary(dataset_final)
View(dataset_final)
str(dataset_final)


# Preparando os dados de treino e de teste
indexes <- sample(1:nrow(dataset_final), size = 0.8 * nrow(dataset_final))
train.data <- dataset_final[indexes,]
test.data <- dataset_final[-indexes,]
class(train.data)
class(test.data)

# Separando os atributos e as classes
test.feature.vars <- test.data[,0:6]
test.class.var <- test.data[,7]
class(test.feature.vars)
test.class.var <- unlist(test.class.var$STATUS) 

# Construindo o modelo de regressão logística
formula.init <- "STATUS ~ ."
formula.init <- as.formula(formula.init)
modelo_v1 <- glm(formula = formula.init, data = train.data, family = "binomial")

# Visualizando os detalhes do modelo
summary(modelo_v1)

# Fazendo previsões e analisando o resultado
previsoes <- predict(modelo_v1, test.data, type = "response")
previsoes <- round(previsoes)
View(previsoes)

# Confusion Matrix
mode(previsoes)
mode(test.class.var)
confusionMatrix(table(data = previsoes, reference = test.class.var), positive = '1')
# Obtemos um resultado de 90% de acuracia para o primeiro modelo que é muito bom, mas vou testar outros modelos

# Vou analisar se conseguimos mudar algo e conseguir um modelo melhor fazendo a Feature Selection
# Feature Selection
formula <- "STATUS ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)

# Plot
plot(importance)

# Construindo um novo modelo com as variáveis de mais importancia, no caso vou descartar as variaveis DESIBEL e FUEL
formula_2 <- "STATUS ~ AIRFLOW + SIZE + DISTANCE"
formula_2 <- as.formula(formula_2)
modelo_v2 <- glm(formula = formula_2, data = train.data, family = "binomial")

# Visualizando o novo modelo
summary(modelo_v2)

# Prevendo e Avaliando o modelo 
previsoes_new <- predict(modelo_v2, test.data, type = "response") 
previsoes_new <- round(previsoes_new)

# Confusion Matrix
confusionMatrix(table(data = previsoes_new, reference = test.class.var), positive = '1')

# Construindo mais um modelo desta vez retirando a variavel SIZE e adicionando a FREQUENCY
formula_3 <- "STATUS ~ AIRFLOW + FREQUENCY + DISTANCE"
formula_3 <- as.formula(formula_3)
modelo_v3 <- glm(formula = formula_3, data = train.data, family = "binomial")

# Visualizando o novo modelo
summary(modelo_v3)

# Prevendo e Avaliando o modelo 
previsoes_3 <- predict(modelo_v3, test.data, type = "response") 
previsoes_3 <- round(previsoes_3)

# Confusion Matrix
confusionMatrix(table(data = previsoes_3, reference = test.class.var), positive = '1')

# Construindo mais um modelo desta vez retirando a variavel FREQUENCY
formula_4 <- "STATUS ~ AIRFLOW + DISTANCE"
formula_4 <- as.formula(formula_4)
modelo_v4 <- glm(formula = formula_4, data = train.data, family = "binomial")

# Visualizando o novo modelo
summary(modelo_v4)

# Prevendo e Avaliando o modelo 
previsoes_4 <- predict(modelo_v4, test.data, type = "response") 
previsoes_4 <- round(previsoes_4)

# Confusion Matrix
confusionMatrix(table(data = previsoes_4, reference = test.class.var), positive = '1')

# Após as tentativas de usar nos modelos diferentes combinações de variaveis o modelo_v1 foi o que alcançou mais acuracia

# Avaliando a performance do modelo escolhido, modelo_v1

# Plot do modelo com melhor acurácia
modelo_final <- modelo_v1
previsoes <- predict(modelo_final, test.feature.vars, type = "response")
previsoes <- round(previsoes)
previsoes_finais <- prediction(previsoes, test.class.var)
confusionMatrix(table(data = previsoes, reference = test.class.var), positive = '1')

# Função para Plot ROC 
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8,xaxs = "i", yaxs = "i")
  abline(0,1, col = "red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
}

# Plot
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais, title.text = "Curva ROC")

# Fazendo previsões em novos dados
# Novos dados
SIZE <- c(1, 4, 3)
FUEL <- c('gasoline', 'gasoline', 'kerosene')
DISTANCE <- c(10, 120, 80)
DESIBEL <- c(75, 96, 101)
AIRFLOW <- c(4.5, 3.2, 2.7)
FREQUENCY <- c(14, 28, 47)

# Cria um dataframe
novo_dataset <- data.frame(SIZE, FUEL, DISTANCE, DESIBEL, AIRFLOW, FREQUENCY)
View(novo_dataset)

# Separa variáveis explanatórias numéricas e categóricas
new.numeric.vars <- c("DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY")

# Aplica as transformações
novo_dataset$FUEL = as.factor(novo_dataset$FUEL)
novo_dataset$SIZE = as.factor(novo_dataset$SIZE)
novo_dataset_final <- scale.features(novo_dataset, new.numeric.vars)
str(novo_dataset_final)
View(novo_dataset_final)

# Previsões
previsao_novo_cliente <- predict(modelo_final, newdata = novo_dataset_final, type = "response")
round(previsao_novo_cliente)

# Vou analisar usando outro modelo de regressao logistica

# Dividindo os dados em treino e teste
split <- initial_split(df, prop = 0.8, strata = STATUS)
train <- split %>% 
  training()
test <- split %>% 
  testing()

# Treinando usando o logistic_reg
model <- logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(STATUS ~ ., data = train)

# Sumario do Modelo
tidy(model)

# Predicoes 
pred_class <- predict(model,
                      new_data = test,
                      type = "class")

# Probabilidades
pred_proba <- predict(model,
                      new_data = test,
                      type = "prob")

results <- test %>%
  select(STATUS) %>%
  bind_cols(pred_class, pred_proba)

accuracy(results, truth = STATUS, estimate = .pred_class)

# Definir o modelo de regressao logistica com penalidades e mix dos hiperparametros
log_reg <- logistic_reg(mixture = tune(), penalty = tune(), engine = "glmnet")
log_reg

# Define o grid de busca para os melhores hiperparametros
grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 4, penalty = 3))
grid

# Define o workflow para o modelo 
log_reg_wf <- workflow() %>%
  add_model(log_reg) %>%
  add_formula(STATUS ~ .)

# Defina o método de reamostragem para a pesquisa no grid
folds <- vfold_cv(train, v = 5)

# Ajustando os hiperparâmetros usando a pesquisa no grid
log_reg_tuned <- tune_grid(
  log_reg_wf,
  resamples = folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

select_best(log_reg_tuned, metric = "roc_auc")

# Treinando o modelo usando o hiperparametros otimizados
log_reg_final <- logistic_reg(penalty = 0.0000000001, mixture = 0.333) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(STATUS ~., data = train)

# Avaliando a performance do modelo no conjunto de testes
pred_class <- predict(log_reg_final,
                      new_data = test,
                      type = "class")
results <- test %>%
  select(STATUS) %>%
  bind_cols(pred_class, pred_proba)

# Criando a confusion matrix
conf_mat(results, truth = STATUS,
         estimate = .pred_class)

# Comparando com o modelo anterior na melhor versão
confusionMatrix(table(data = previsoes, reference = test.class.var), positive = '1')

# A acuracia do modelo usando glm() foi melhor do que do que o segundo modelo usado.
# Accuracy 90.6%
