# Projeto da Formação Cientista de Dados da Data Science Academy

## Machine Learning na Segurança do Trabalho Prevendo a Eficiência de Extintores de Incêndio
<p align="justify">O teste hidrostático extintor é um procedimento estabelecido pelas normas da ABNT
NBR 12962/2016, que determinam que todos os extintores devem ser testados a cada cinco
anos, com a finalidade de identificar eventuais vazamentos, além de também verificar a
resistência do material do extintor.</p>
<p align="justify">Com isso, o teste hidrostático extintor pode ser realizado em baixa e alta pressão, de
acordo com estas normas em questão. O procedimento é realizado por profissionais técnicos
da área e com a utilização de aparelhos específicos e apropriados para o teste, visto que eles
devem fornecer resultados com exatidão.</p>
<p align="justify">Seria possível usar Machine Learning para prever o funcionamento de um extintor de
incêndio com base em simulações feitas em computador e assim incluir uma camada adicional
de segurança nas operações de uma empresa?</p>
<p align="justify">O conjunto de dados foi obtido como resultado dos testes de extinção de quatro chamas
de combustíveis diferentes com um sistema de extinção de ondas sonoras. O sistema de extinção
de incêndio por ondas sonoras consiste em 4 subwoofers com uma potência total de 4.000 Watts.
Existem dois amplificadores que permitem que o som chegue a esses subwoofers como
amplificado. A fonte de alimentação que alimenta o sistema e o circuito do filtro garantindo que
as frequências de som sejam transmitidas adequadamente para o sistema está localizada dentro
da unidade de controle. Enquanto o computador é usado como fonte de frequência, o
anemômetro foi usado para medir o fluxo de ar resultante das ondas sonoras durante a fase de
extinção da chama e um decibelímetro para medir a intensidade do som. Um termômetro
infravermelho foi utilizado para medir a temperatura da chama e da lata de combustível, e uma
câmera é instalada para detectar o tempo de extinção da chama. Um total de 17.442 testes foram
realizados com esta configuração experimental. Os experimentos foram planejados da seguinte
forma:</p>
<p align="justify">&nbsp;&nbsp;&nbsp;* Três diferentes combustíveis líquidos e combustível GLP foram usados para criar a
chama.</p>
<p align="justify">&nbsp;&nbsp;&nbsp;* 5 tamanhos diferentes de latas de combustível líquido foram usados para atingir
diferentes tamanhos de chamas.</p>
<p align="justify">&nbsp;&nbsp;&nbsp;* O ajuste de meio e cheio de gás foi usado para combustível GLP.</p>
<p align="justify">Durante a realização de cada experimento, o recipiente de combustível, a 10 cm de
distância, foi movido para frente até 190 cm, aumentando a distância em 10 cm a cada vez. Junto
com o recipiente de combustível, o anemômetro e o decibelímetro foram movidos para frente
nas mesmas dimensões.</p>
<p align="justify">Experimentos de extinção de incêndio foram conduzidos com 54 ondas sonoras de
frequências diferentes em cada distância e tamanho de chama.
Ao longo dos experimentos de extinção de chama, os dados obtidos de cada dispositivo
de medição foram registrados e um conjunto de dados foi criado. O conjunto de dados inclui as
características do tamanho do recipiente de combustível representando o tamanho da chama,
tipo de combustível, frequência, decibéis, distância, fluxo de ar e extinção da chama. Assim, 6
recursos de entrada e 1 recurso de saída serão usados no modelo que você vai construir.
A coluna de status (extinção de chama ou não extinção da chama) pode ser prevista
usando os seis recursos de entrada no conjunto de dados. Os recursos de status e combustível
são categóricos, enquanto outros recursos são numéricos.</p>
