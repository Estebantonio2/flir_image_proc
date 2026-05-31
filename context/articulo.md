Universidad de Lima

Facultad de Ingeniería

Carrera de Ingeniería de Sistemas

![](data:image/png;base64...)

**MODELO DE VISIÓN COMPUTACIONAL ADAPTATIVO PARA LA ESTIMACIÓN TEMPORAL DE RASTROS TÉRMICOS DE LA MANO EN SUPERFICIES CONTROLADAS**

Trabajo de investigación para optar el Título Profesional de Ingeniero de Sistemas

**Esteban Antonio Terrones Torres**

**Código 20223007**

**Asesor**

Edwin Jonathan Escobedo Cardenas

Lima – Perú

Mayo de 2026

**Modelo de visión computacional adaptativo para la estimación temporal de rastros térmicos de la mano en superficies controladas**

**Esteban Antonio Terrones Torres**

20223007@aloe.ulima.edu.pe

Universidad de Lima

**INTRODUCCIÓN**

En 2023, la Oficina de las Naciones Unidas contra la Droga y el Delito (UNODC) reportó aproximadamente 458,000 homicidios en el mundo, advirtiendo una tendencia al incremento de la criminalidad en los próximos años (UNODC, 2023, p. 8). En el Perú, el Instituto Nacional de Estadística e Informática (INEI) registró cerca de 3,000 muertes violentas vinculadas a delitos dolosos en 2021, un aumento aproximado del 50% respecto al año anterior (INEI, 2023, p. 22). A ello se suma que en las Américas es menos frecuente que los casos de homicidio lleguen a resolverse completamente, lo que evidencia una brecha en la efectividad de los sistemas de investigación penal (UNODC, 2023, p. 148).

La reconstrucción de escenas del crimen busca identificar y preservar evidencias para esclarecer las circunstancias de un delito (Esposito et al., 2023), pero enfrenta limitaciones como la contaminación o ausencia de evidencia (Lloyd Institute of Forensic Science, n.d.). En este contexto, la imagen térmica surge como una técnica no destructiva aplicable a la investigación forense (Hou et al., 2022) que, a diferencia de las huellas dactilares tradicionales, permite obtener evidencia incluso cuando el sujeto usa guantes (Cho et al., 2016). Las deficiencias en estos procesos afectan principalmente a víctimas y sus familias, generando retrasos en la justicia y mayor percepción de impunidad (Wickenheiser, 2023).

La literatura sobre termografía forense muestra una evolución desde modelos físicos hasta sistemas basados en Deep Learning. Xu et al. (2020) estimaron el tiempo de partida a partir de huellas térmicas aplicando la ley de enfriamiento de Newton, con correlaciones superiores al 99%, aunque con pérdida de precisión a mayor tiempo transcurrido. Wilk et al. (2021) combinaron fotogrametría térmica con un modelo termodinámico para estimar el intervalo post mortem, logrando una precisión de 16 minutos. Con el aprendizaje profundo, Yu, Liang, Zhou, Zhang, et al. (2024) desarrollaron un modelo híbrido basado en ResNet que alcanzó 80.45% en reconocimiento de identidad y 94.29% en estimación temporal. Yu, Liang, Zhou & Zhang (2024) mejoraron estos resultados al 83.48% en identificación con residuales menores a 120 segundos.

Sin embargo, estos modelos tratan los factores ambientales de forma limitada. Dar et al. (2022) señalan que la temperatura ambiental influye directamente en la disipación térmica, y Wilk et al. (2021) junto con Xu et al. (2020) demuestran que el flujo de aire y las propiedades del material también afectan la precisión de las curvas térmicas, pero la mayoría asume condiciones constantes o trabaja en entornos controlados. Yu, Liang, Zhou & Zhang (2024) aplican calibraciones periódicas sin lograr adaptación en tiempo real. En otros campos, integrar variables ambientales en modelos multimodales ha mejorado notablemente el rendimiento, como en reconocimiento de emociones (Kim & Hong, 2024), predicción de calidad del aire (Lilhore et al., 2025) y compensación térmica en imágenes infrarrojas (Y. Li et al., 2024).

En base a esto, se propone desarrollar un modelo de visión computacional adaptativo para estimar el tiempo transcurrido desde el contacto de una mano sobre superficies controladas, integrando imágenes termográficas con temperatura ambiente y humedad relativa. El estudio busca abordar las limitaciones asociadas a la influencia del entorno en la disipación térmica, señaladas por Xu et al. (2020), Wilk et al. (2021) y Dar et al. (2022), mediante la construcción de un dataset propio y el entrenamiento de arquitecturas de Deep Learning multimodales. A partir de ello, se plantea la pregunta de investigación: ¿cómo la integración de variables ambientales mejora la precisión de los modelos de Deep Learning para estimar el tiempo de disipación de rastros térmicos de la mano sobre superficies? Para responderla, se identifican variables relevantes, se caracterizan modelos y métricas, se construye el dataset, se entrenan modelos multimodales y se valida su desempeño mediante regresión y pruebas de tolerancia temporal.

El resto del artículo se organiza en cuatro secciones. Primero, se revisan los trabajos relacionados para identificar avances y limitaciones del campo. Luego, se presentan los fundamentos teóricos de la disipación térmica y el modelamiento multimodal. Posteriormente, se describe la metodología propuesta y, finalmente, se exponen los avances experimentales obtenidos en la adquisición y preparación del dataset.

**TRABAJOS RELACIONADOS**

La presente sección revisa la literatura relacionada con el análisis de rastros térmicos y el uso de modelos de aprendizaje automático y Deep Learning en imágenes termográficas. La discusión se organiza en cinco líneas temáticas: el uso de la disipación térmica como señal temporal, los datasets y condiciones experimentales de adquisición, las estrategias de preprocesamiento y representación de imágenes térmicas, los modelos de Deep Learning y enfoques espacio-temporales, y las métricas empleadas para evaluar su desempeño. Esta organización permite comparar estudios con distintos niveles de cercanía al problema, diferenciando entre trabajos directamente orientados a huellas térmicas, investigaciones metodológicamente transferibles y aplicaciones contextuales en otros dominios termográficos. A partir de esta comparación, se identifican tendencias, limitaciones metodológicas y vacíos que permiten posicionar el presente estudio dentro del campo.

**Rastros térmicos como señal temporal**

La literatura reciente coincide en que los rastros térmicos no deben entenderse solo como residuos de calor, sino como señales temporales que conservan información sobre interacciones previas. Esta idea se observa en trabajos que reconstruyen estados pasados de una escena o infieren acciones humanas a partir de calor residual, como Tang et al. (2023) y Contreras et al. (2025). Sin embargo, estos enfoques se diferencian del presente estudio porque priorizan la reconstrucción visual o contextual del evento, mas no la estimación numérica del tiempo transcurrido desde un contacto específico.

Los estudios más próximos al problema abordan la disipación térmica como variable temporal directa, aunque desde enfoques distintos. Xu et al. (2020) representan el decaimiento térmico mediante un modelo físico basado en la Ley de Enfriamiento de Newton, mientras que Yu, Liang, Zhou, & Zhang (2024) y Yu, Liang, Zhou, Zhang, et al. (2024) emplean modelos de Deep Learning como MTLHand y DSTFS para aprender patrones asociados al desvanecimiento, pérdida de contraste y desenfoque de huellas térmicas de mano. En contraste, otros trabajos aprovechan los rastros térmicos con fines complementarios, como detección de puntos de contacto, segmentación de huellas difusas, caracterización de materiales o mitigación de marcas residuales como interferencia (Ma et al., 2021; Zhou et al., 2022; Dar et al., 2022; Lee et al., 2022).

En conjunto, la literatura muestra una convergencia clara en torno al valor informativo de la disipación térmica, pero también una divergencia en su uso metodológico. Mientras algunos estudios la emplean para reconstruir, segmentar o detectar eventos, pocos la orientan directamente a la estimación temporal de huellas de mano. Esta comparación evidencia la necesidad de integrar el comportamiento físico del rastro, las condiciones de adquisición y modelos capaces de estimar el tiempo transcurrido bajo escenarios controlados.

**Datasets y condiciones experimentales**

La literatura revisada muestra una disponibilidad limitada de datasets públicos orientados específicamente al análisis temporal de rastros térmicos humanos. Aunque conjuntos abiertos como FLIR Thermal o KAIST Multispectral son ampliamente usados en imágenes térmicas, su finalidad principal es la detección de objetos, escenas o peatones, por lo que no incorporan necesariamente contacto humano, disipación residual ni etiquetas temporales asociadas al rastro (Raj et al., 2025; Sachan et al., 2022). En cambio, los datasets más cercanos al problema suelen ser propios o de acceso restringido: IRHTv2c6 se orienta a huellas térmicas de mano e identidad, HT\_IC y HT\_DF a segmentación de rastros difusos, y Thermal-IM a inferencia de pose con datos RGB-T-D, pero ninguno integra de forma amplia contacto humano, diversidad de superficies, regresión temporal y variables ambientales en un mismo protocolo (Yu, Liang, Zhou & Zhang, 2024; Zhou et al., 2022; Tang et al., 2023).

Respecto a las condiciones experimentales, los estudios revisados también presentan diferencias en el nivel de control y reporte. Algunos documentan superficie, tiempo de contacto, cámara, distancia o frecuencia de captura, mientras que variables como humedad relativa, emisividad, presión de contacto o flujo de aire suelen omitirse o asumirse constantes. Esta omisión contrasta con estudios de caracterización física, donde el material, el tiempo de contacto y las condiciones ambientales modifican la curva de disipación térmica (Dar et al., 2022; Ai et al., 2020). Además, en dominios médicos e industriales, estas variables sí han sido incorporadas como características del modelo, lo que sugiere su utilidad para mejorar la estimación bajo condiciones variables (Razmara et al., 2024; Holgado-Apaza et al., 2025).

En conjunto, los datasets disponibles evidencian avances importantes, pero también una fragmentación metodológica: algunos priorizan detección, otros segmentación, otros identidad o reconstrucción, y pocos se orientan directamente a la datación de rastros térmicos de mano. Esta comparación muestra que la principal limitación no es solo la escasez de datos, sino la falta de protocolos que integren imágenes termográficas, etiquetas temporales, superficies diferenciadas y variables ambientales. Por ello, se justifica la construcción de un dataset controlado y multimodal para estudiar la disipación térmica bajo condiciones experimentales trazables.

**Preprocesamiento de imágenes térmicas**

El preprocesamiento en imágenes térmicas es una etapa crítica porque la señal infrarroja suele presentar bajo contraste, ruido del sensor y pérdida progresiva de nitidez conforme avanza la disipación. La literatura coincide en aplicar normalización, escalado, reducción de ruido, sustracción de fondo, redimensionamiento y selección de regiones de interés para estabilizar la entrada del modelo y resaltar cambios térmicos relevantes (Boiko et al., 2022; Ma et al., 2021; Mentzel et al., 2021; Sachan et al., 2022; St-Antoine et al., 2025). Sin embargo, estos procedimientos difieren en su capacidad para preservar información temporal: mientras los filtros genéricos mejoran la calidad visual, enfoques como DSTFS incorporan umbral suave profundo para separar de forma aprendida información temporal, identidad y ruido térmico en rastros de bajo contraste (Yu, Liang, Zhou, Zhang, et al., 2024).

Una segunda diferencia se observa en la forma de aislar y representar la señal. Algunos estudios emplean recortes manuales o regiones de interés para concentrar el análisis en la zona térmicamente relevante, mientras que otros utilizan segmentación automática mediante arquitecturas como U-Net o ICDNet cuando el rastro se confunde con el fondo (Ma et al., 2021; Zhou et al., 2022; Yu, Liang, Zhou & Zhang, 2024). De forma similar, el redimensionamiento facilita el uso de arquitecturas convolucionales, pero puede reducir detalles térmicos finos si la resolución disminuye en exceso (Lee et al., 2022; Tang et al., 2025; Zhao et al., 2024). En la representación temporal, las curvas de decaimiento simplifican la regresión, aunque sacrifican información espacial, mientras que las secuencias, mapas de diferencia térmica y vectores de contraste conservan mejor la dinámica de enfriamiento (Xu et al., 2020; Garrido et al., 2021; Lee et al., 2022; Strąkowska & Strzelecki, 2023).

En otros trabajos, la reducción de dimensionalidad, codificación de variables e ingeniería de características permiten compactar información térmica y contextual antes del modelamiento (Boiko et al., 2022; Hao et al., 2023; Movahedi-Rad & Keller, 2026; Razmara et al., 2024; Xiao & Chen, 2024). No obstante, en estudios sobre rastros térmicos humanos, las variables no visuales, como temperatura ambiente, humedad o propiedades de superficie, suelen tratarse como condiciones experimentales y no como entradas activas del modelo. En síntesis, la comparación muestra que el preprocesamiento no solo debe limpiar la imagen, sino definir qué información se conserva para la estimación temporal: intensidad térmica, forma del rastro, cambio entre capturas y contexto ambiental asociado.

**Modelos espacio-temporales de Deep Learning**

Los modelos de Deep Learning aplicados a imágenes térmicas muestran distintas formas de aprovechar la información contenida en los rastros. En los estudios más cercanos al problema, MTLHand y DSTFS emplean arquitecturas basadas en ResNet para separar características asociadas a identidad y tiempo, abordando el desvanecimiento y desenfoque progresivo de las huellas térmicas (Yu, Liang, Zhou & Zhang, 2024; Yu, Liang, Zhou, Zhang, et al., 2024). En cambio, ICDNet prioriza la segmentación de huellas difusas, mientras que Tang et al. (2023) y Contreras et al. (2025) utilizan la señal térmica residual para inferir eventos o estados pasados. En conjunto, estos trabajos coinciden en que los rastros térmicos contienen información espacial y temporal útil, aunque difieren en la tarea abordada, ya que no todos formulan el problema como estimación temporal directa (Zhou et al., 2022).

Una segunda línea se orienta a representar la evolución térmica como una dinámica temporal. Modelos como GRU, BiLSTM, CRNN y ConvLSTM permiten analizar la señal como una secuencia, lo cual resulta pertinente cuando la intensidad térmica debe interpretarse como parte de una curva de disipación y no como una medición aislada (Boiko et al., 2022; Garrido et al., 2021; Movahedi-Rad & Keller, 2026; Sajadi et al., 2024). A diferencia de estos enfoques puramente aprendidos, modelos físicamente informados como PI-ConvLSTM incorporan restricciones de transferencia de calor para mejorar la consistencia de la predicción en campos térmicos dinámicos (Sajadi et al., 2024). Asimismo, estudios sobre curvas térmicas, flujos de calor y regresión de propiedades físicas muestran que las redes neuronales pueden capturar relaciones no lineales entre temperatura, tiempo y comportamiento del material (Hao et al., 2023; H. Li et al., 2024; Strąkowska & Strzelecki, 2023; Xiao & Chen, 2024).

También existen modelos térmicos de detección, segmentación, clasificación y regresión aplicados a dominios forenses, médicos, industriales o ambientales que aportan evidencia metodológica, aunque no resuelven directamente la datación de huellas térmicas. Detectores como YOLOv5, Faster R-CNN o RetinaNet son útiles para localizar objetos o regiones con contraste térmico, pero no necesariamente para estimar señales residuales que se degradan gradualmente (Lee et al., 2022; Raj et al., 2025; Sachan et al., 2022). De forma complementaria, CNN especializadas y modelos híbridos han sido aplicados en detección forense, monitoreo de baterías, control térmico fotovoltaico, estimación de gases y predicción de variables físicas o fisiológicas (Brezov et al., 2023; Elmessery et al., 2024; Holgado-Apaza et al., 2025; J. & Thinakaran, 2023; Maino et al., 2024; Razmara et al., 2024; Senthilraj & Shanker, 2023; Tang & Sato, 2026; Tang et al., 2025).

En síntesis, la literatura confirma el potencial predictivo de la termografía con Deep Learning, pero también evidencia una separación entre modelos espaciales, secuenciales, físicamente informados y multimodales. Para la estimación temporal de rastros térmicos, esta fragmentación sugiere que no basta con detectar o clasificar la huella, sino que se requiere integrar extracción espacial, evolución temporal y variables contextuales que condicionan la disipación.

**Métricas y brecha de investigación**

Las métricas empleadas en la literatura varían según la tarea abordada, por lo que los resultados no son comparables de forma directa entre segmentación, reconstrucción, detección y estimación temporal. En segmentación predominan métricas como mIoU o Dice, orientadas a medir el solapamiento espacial del rastro, como ocurre en ICDNet para huellas térmicas difusas (Zhou et al., 2022). En reconstrucción de eventos se emplean métricas como MPJPE, PSNR o SSIM, usadas en Thermal-IM y en enfoques de reconstrucción inversa de escenas (Tang et al., 2023; Contreras et al., 2025). En cambio, la estimación temporal requiere medir error cronológico mediante Error-60, Error-120, Accuracy-60, Accuracy-120 o métricas de regresión como MAE, RMSE, MSE, MAPE y R². En este grupo, MTLHand y DSTFS son los referentes más cercanos para huellas térmicas de mano, aunque sus resultados deben interpretarse según sus propios datasets, superficies y protocolos de captura (Yu, Liang, Zhou & Zhang, 2024; Yu, Liang, Zhou, Zhang, et al., 2024).

En tareas térmicas dinámicas o de regresión física también se reportan desempeños relevantes, pero su transferencia al problema de datación de huellas es parcial. Modelos como PI-ConvLSTM, CRNN y BiLSTM muestran capacidad para procesar campos o secuencias térmicas, mientras que CNN 1D, BPNN, Gradient Boost y otros modelos de regresión han sido aplicados a curvas térmicas, flujos de calor, propiedades físicas, enfriamiento de paneles, monitoreo de baterías y concentración de gases (Boiko et al., 2022; Movahedi-Rad & Keller, 2026; Sajadi et al., 2024; Elmessery et al., 2024; Hao et al., 2023; Holgado-Apaza et al., 2025; H. Li et al., 2024; Mentzel et al., 2021; Senthilraj & Shanker, 2023; Strąkowska & Strzelecki, 2023; Xiao & Chen, 2024). Sin embargo, estos resultados responden a variables objetivo, escalas temporales y condiciones experimentales distintas. De igual modo, los avances en detección térmica o clasificación forense evidencian capacidad de análisis infrarrojo, pero no resuelven la estimación temporal de una señal residual que se degrada progresivamente (J. & Thinakaran, 2023; Raj et al., 2025; Sachan et al., 2022).

En conjunto, la literatura confirma que los rastros térmicos pueden ser útiles para segmentación, detección, reconstrucción, clasificación y estimación temporal, pero también evidencia una brecha metodológica. Persisten tres limitaciones principales: la falta de criterios estandarizados para comparar el error temporal continuo, la dependencia de datasets pequeños o específicos y la escasa integración de variables ambientales y características de superficie en modelos de datación. Aunque MTLHand, DSTFS y los modelos físicos de Xu et al. (2020) avanzan hacia la estimación temporal, todavía se requiere evaluar cómo interactúan la señal residual, el material, el ambiente y las condiciones de adquisición. Esta brecha justifica el desarrollo de un enfoque que articule imágenes termográficas, variables experimentales y métricas de regresión temporal bajo un protocolo controlado.

**MARCO TEÓRICO**

La presente sección aborda los fundamentos teóricos necesarios para comprender la estimación del tiempo transcurrido desde la formación de una huella térmica de mano sobre una superficie. Para ello, se consideran el comportamiento físico del rastro térmico, su evolución temporal, las condiciones de adquisición termográfica y el uso de modelos de Deep Learning multimodal para integrar información visual y ambiental. Estos fundamentos permiten entender que la imagen termográfica no debe interpretarse como una señal aislada, sino como el resultado de una interacción entre transferencia de calor, propiedades del material, condiciones ambientales y características del sensor.

**Fundamentos de los rastros térmicos**

Un rastro térmico puede definirse como la radiación infrarroja residual que permanece en una superficie después de una interacción física (Tang et al., 2023). Su formación se explica por la diferencia de temperatura entre la piel y la superficie de contacto, ya que los seres humanos son organismos homeotermos, la temperatura cutánea suele diferir de la de los objetos del entorno, generando un gradiente térmico que impulsa la transferencia de energía durante el contacto (Kaczmarek et al., 2018; Zhou et al., 2022). En este proceso, el mecanismo predominante es la conducción, mediante la cual el calor se transfiere desde la piel hacia el material receptor. Este proceso depende de la diferencia de temperatura, el área de contacto, la presión ejercida, el tiempo de permanencia y las propiedades termofísicas de la superficie, como la conductividad térmica, la capacidad calorífica y la difusividad térmica (Kaczmarek et al., 2018; Luo et al., 2019; Zhou et al., 2022). Una vez retirada la mano, el calor residual se redistribuye dentro del material y se disipa hacia el entorno mediante convección y radiación, hasta alcanzar el equilibrio térmico con el ambiente (Dar et al., 2022; Zhou et al., 2022).

La persistencia del rastro depende tanto del tiempo transcurrido como del material sobre el cual se forma. En particular, superficies con distintas propiedades térmicas pueden retener o disipar el calor a velocidades diferentes, modificando la intensidad, el contraste y la nitidez de la huella. A su vez, variables como la emisividad, la temperatura ambiente, la humedad relativa y el flujo de aire condicionan la visibilidad del rastro y su evolución temporal (Cho et al., 2016; Dar et al., 2022; Kaczmarek et al., 2018), por lo que una misma huella puede presentar apariencias térmicas distintas según la superficie y las condiciones ambientales de registro. Conforme avanza el tiempo, la huella pierde intensidad, sus bordes se difuminan y su contraste disminuye. Cuando la señal se aproxima a la temperatura del entorno, el rastro puede volverse difícil de distinguir del ruido térmico de fondo (Xu et al., 2020; Zhou et al., 2022), lo que representa una limitación física relevante que la estimación temporal debe considerar.

**Estimación temporal de señales térmicas**

La estimación temporal de señales térmicas se fundamenta en la relación entre el tiempo transcurrido desde el contacto y la disipación del calor residual. En términos físicos, este comportamiento puede aproximarse mediante la Ley de Enfriamiento de Newton, según la cual la tasa de pérdida de calor es proporcional a la diferencia de temperatura entre el cuerpo y el entorno (Kaczmarek et al., 2018; Xu et al., 2020). Esta relación suele expresarse como:

$T(t)=T\_{s}+(T\_{0}-T\_{s})e^{-κt}$

donde $T(t)$ representa la temperatura del rastro en el tiempo $t$, $T\_{s} $la temperatura del entorno, $T\_{0}$ la temperatura inicial del rastro y $κ$ la constante de enfriamiento. Esta ecuación muestra que la disipación no es lineal, pues al inicio el enfriamiento ocurre con mayor rapidez, y la tasa de cambio disminuye conforme la señal se aproxima al equilibrio térmico (Kaczmarek et al., 2018; Xu et al., 2020).

Sin embargo, una huella térmica real no siempre cumple los supuestos ideales de este modelo. La Ley de Enfriamiento de Newton asume condiciones ambientales constantes, enfriamiento uniforme y propiedades homogéneas del cuerpo analizado. En una huella de mano, estos supuestos pueden verse afectados por la heterogeneidad del material, la distribución irregular de presión, el contacto incompleto, la variación en la temperatura inicial de la piel y la difusión lateral del calor (Xu et al., 2020; Zhou et al., 2022). Además, a medida que la huella se aproxima a la temperatura ambiente, la relación señal-ruido disminuye y la información disponible se vuelve más ambigua, lo que aumenta el error en ventanas temporales prolongadas, cuando los rasgos térmicos dejan de diferenciarse claramente del fondo (Xu et al., 2020; Zhou et al., 2022). Por tanto, la estimación debe entenderse como una tarea condicionada por la calidad de la señal, la superficie de contacto y las condiciones ambientales.

La estimación temporal puede formularse como un problema de regresión, donde la variable objetivo es el tiempo transcurrido y las variables observables provienen de la imagen térmica y del contexto experimental. Estas variables pueden incluir la intensidad de los píxeles, el contraste respecto al fondo, la forma visible de la huella, la difusión de bordes y, en secuencias, la evolución térmica entre capturas sucesivas (Yu, Liang, Zhou & Zhang, 2024; Yu, Liang, Zhou, Zhang, et al., 2024). Esta formulación permite superar la dependencia de una única medición térmica y analizar patrones espaciales y temporales asociados a la disipación.

**Adquisición termográfica**

La termografía infrarroja permite capturar la radiación emitida por los objetos en el espectro infrarrojo y convertirla en una representación electrónica interpretable (Bekhit & Reimert, 2025; Yu et al., 2020). Conviene diferenciar entre una imagen térmica visual y un dato radiométrico, donde la primera corresponde a una representación coloreada de la temperatura aparente, mientras que los datos radiométricos permiten obtener matrices de temperatura o intensidad térmica útiles para el análisis cuantitativo de la disipación (Golosov & Cervone, 2024). Esta distinción es relevante porque la estimación temporal requiere analizar variaciones térmicas graduales, no solo la apariencia visual del rastro.

La temperatura registrada por una cámara termográfica no depende únicamente de la temperatura superficial del objeto, pues la radiación recibida puede incluir emisión propia del objeto, radiación reflejada del entorno y atenuación atmosférica. Factores como la emisividad, la temperatura reflejada, la distancia al objetivo, el ángulo de captura, la resolución espacial, la sensibilidad térmica y la estabilidad del sensor pueden afectar la medición (Mazdeyasna et al., 2023; Usamentiaga et al., 2014). En superficies como el vidrio la reflexión térmica puede ser particularmente relevante, mientras que en materiales como la madera la textura y las propiedades térmicas influyen en la distribución del calor. La sensibilidad del sensor resulta especialmente crítica en los minutos finales, cuando la diferencia entre la huella y el fondo puede ser mínima y la señal puede confundirse con el ruido térmico, afectando la extracción de características. La distancia y el ángulo de captura también deben controlarse, ya que modifican la cantidad de píxeles de la huella y la intensidad aparente de la radiación detectada (Mazdeyasna et al., 2023).

Además, la temperatura ambiente y la humedad relativa deben registrarse como variables contextuales, dado que influyen en la disipación del calor y en la transmitancia de la señal infrarroja hacia el sensor (J.-Q. Li et al., 2024). En consecuencia, un protocolo de captura consistente es necesario para reducir sesgos y garantizar que las diferencias observadas entre imágenes correspondan principalmente a cambios del rastro térmico, y no a variaciones no controladas del entorno, del sensor o del procedimiento experimental (Nikolov et al., 2021).

**Modelamiento multimodal con Deep Learning**

El Deep Learning permite aprender patrones espaciales y temporales en imágenes termográficas, como la forma del rastro, la intensidad residual, la pérdida de contraste, la difusión de bordes y la evolución térmica entre capturas (Luo et al., 2019; Yu, Liang, Zhou & Zhang, 2024). Las redes convolucionales, como ResNet o EfficientNet, son adecuadas para extraer características espaciales, mientras que modelos secuenciales como LSTM o ConvLSTM permiten representar dependencias temporales en secuencias de imágenes (He et al., 2015; Shi et al., 2015; Yu, Liang, Zhou, Zhang, et al., 2024). Diversos estudios han evidenciado la utilidad de estos enfoques aplicados a rastros térmicos: redes convolucionales profundas han demostrado capacidad para clasificar handprints infrarrojos incluso en imágenes con bajo contraste (Zhou et al., 2021); arquitecturas como ICDNet abordan específicamente la segmentación de rastros difusos (Zhou et al., 2022), mientras que DSTFS incorpora mecanismos para separar características de identidad, tiempo y ruido térmico (Yu, Liang, Zhou, Zhang, et al., 2024). Estos desarrollos confirman que los rastros térmicos contienen información visual útil pero susceptible de degradación, ruido y variabilidad experimental.

La multimodalidad resulta pertinente porque la apariencia de una huella térmica no depende únicamente del tiempo transcurrido, sino también de las condiciones físicas bajo las cuales se produce la disipación. La temperatura ambiente actúa como referencia de equilibrio térmico, mientras que la humedad relativa puede influir en la transmitancia atmosférica y en la medición de la señal infrarroja (J.-Q. Li et al., 2024; Mazdeyasna et al., 2023). Por ello, integrar variables ambientales permite que el modelo no interprete toda reducción de intensidad como efecto exclusivo del tiempo, sino que considere el contexto que condiciona la pérdida de calor.

Existen distintas estrategias de fusión multimodal, donde la temprana integra las modalidades desde la entrada, la intermedia combina representaciones latentes de ramas especializadas, y la tardía une características de alto nivel antes de la salida. Para datos heterogéneos como imágenes térmicas y variables numéricas ambientales, una estrategia de ramas separadas resulta defendible, pues una CNN procesa la imagen térmica, mientras que una red densa o MLP transforma la temperatura ambiente y la humedad en una representación numérica antes de fusionarse con las características visuales (Golosov & Cervone, 2024). Sin embargo, este enfoque presenta limitaciones como mayor complejidad del modelo, riesgo de sobreajuste en datasets reducidos y dependencia de la calidad de los sensores ambientales.

Los fundamentos revisados permiten comprender que la estimación temporal de huellas térmicas no depende únicamente de la intensidad visible del rastro, sino de la interacción entre transferencia de calor, propiedades de la superficie, condiciones ambientales, adquisición radiométrica y capacidad del modelo para aprender patrones de disipación. Por ello, un enfoque basado en Deep Learning multimodal resulta adecuado para estimar el tiempo transcurrido desde el contacto, siempre que se controlen las condiciones de captura, se registren variables ambientales relevantes y se reconozca el límite físico impuesto por la pérdida progresiva de contraste térmico.

**METODOLOGÍA**

La presente investigación adopta un enfoque cuantitativo, aplicado y experimental, debido a que busca desarrollar un modelo de Deep Learning para estimar el tiempo transcurrido desde la formación de rastros térmicos de la mano sobre superficies, integrando imágenes termográficas y variables ambientales. La contribución académica consiste en evaluar la incorporación de temperatura ambiente y humedad relativa como entradas complementarias para la estimación temporal, dado que los trabajos revisados evidencian limitaciones en la disponibilidad de datasets específicos, diversidad de superficies y registro explícito de condiciones ambientales en modelos de datación térmica. El estudio se desarrollará en un entorno de laboratorio, mediante un protocolo de captura con participantes voluntarios, superficies controladas, secuencias térmicas y registros ambientales asociados.

Para la implementación se utilizarán técnicas de adquisición termográfica, preprocesamiento de imágenes térmicas, extracción de matrices radiométricas, generación de variables temporales y entrenamiento de arquitecturas de Deep Learning. Las herramientas principales incluyen una aplicación móvil desarrollada en Android/Kotlin con el FLIR Mobile SDK para automatizar la captura, la librería FlirImageExtractor para obtener matrices térmicas a partir de las imágenes, Python para el procesamiento de datos y PyTorch para la implementación de los modelos. La solución será evaluada mediante métricas de regresión y tolerancia temporal, como MAE, RMSE, R², Accuracy-60 y Accuracy-120, con el fin de analizar el error de estimación y obtener resultados iniciales. El flujo metodológico comprende cinco etapas: diseño experimental y sistema de adquisición, construcción del dataset base, preprocesamiento y generación de entradas, implementación y entrenamiento de modelos, y validación preliminar con análisis de resultados iniciales.

**Figura 1**

*Pipeline metodológico propuesto*

![](data:image/png;base64...)

La metodología se organiza en cinco etapas secuenciales. Primero, se define el diseño experimental y el sistema de adquisición. Luego, se construye el dataset base a partir de secuencias térmicas y registros ambientales. Posteriormente, se realiza el preprocesamiento y la generación de entradas para los modelos. Después, se implementan y entrenan las arquitecturas de estimación temporal. Finalmente, se desarrolla una validación preliminar orientada a obtener resultados iniciales. Este pipeline puede visualizarse en la Figura 1.

**Etapa 1. Diseño experimental y sistema de adquisición**

En la actual etapa se define el protocolo experimental y se desarrolla el sistema de adquisición para recolectar secuencias térmicas de forma automatizada. Esta etapa busca asegurar que las capturas mantengan condiciones consistentes de registro, considerando que la estimación temporal depende de la calidad de la señal, la superficie de contacto, las condiciones ambientales, la distancia y el ángulo de captura. Además, permite organizar el procedimiento antes de iniciar la construcción del dataset, siguiendo la necesidad metodológica de definir instrumentos, técnicas y procedimientos de recolección de datos.

1. Definición del protocolo experimental: Se define un protocolo experimental para la adquisición de secuencias térmicas generadas por contacto de la mano sobre superficies controladas. El protocolo considera participantes voluntarios, superficies de prueba, tiempo de contacto, duración de captura, distancia cámara-superficie, posición de adquisición, condiciones ambientales y registro de metadatos. Estos parámetros serán especificados y reportados en la sección de experimentación, de acuerdo con el alcance de la fase desarrollada.
2. Selección de instrumentos de adquisición: Se emplea una cámara FLIR One Pro para capturar imágenes térmicas y un termohigrómetro Elitech RC-51H para registrar temperatura ambiente y humedad relativa durante cada secuencia. La cámara permite observar la evolución del rastro térmico, mientras que el termohigrómetro permite registrar variables contextuales que pueden afectar la disipación y la lectura infrarroja. Esta selección responde a la necesidad de integrar información visual y ambiental en el estudio.
3. Desarrollo de la aplicación móvil Android/Kotlin: Se implementa una aplicación móvil en Android/Kotlin mediante el FLIR Mobile SDK para automatizar la captura secuencial de imágenes térmicas. Esta herramienta reduce la dependencia de capturas manuales, favorece la regularidad temporal entre tomas y organiza los archivos generados por cada secuencia.
4. Pruebas piloto y ajuste operativo: Se realizan pruebas iniciales para verificar el funcionamiento de la aplicación, la estabilidad de captura, el formato de almacenamiento, la visibilidad de la huella y la autonomía real de la cámara. Estos ensayos permiten ajustar la planificación de sesiones, considerando que la batería de la FLIR One Pro limita la cantidad de participantes por jornada. Esta actividad reduce errores antes de la recolección principal y mejora la replicabilidad del protocolo.

**Etapa 2. Construcción y preparación del dataset base**

En la actual etapa se recolectan las secuencias térmicas y se construye un dataset base estandarizado, que servirá como punto de partida para generar las entradas específicas de los modelos. Esta etapa es central porque la literatura evidencia una disponibilidad limitada de datasets públicos orientados a huellas térmicas de mano, y varios estudios cercanos emplean conjuntos propios o de acceso restringido, como IRHTv2c6, HT\_IC o HT\_DF (Yu, Liang, Zhou, Zhang, et al., 2024; Yu, Liang, Zhou & Zhang, 2024; Zhou et al., 2022). Además, se busca integrar imágenes termográficas, etiquetas temporales y variables ambientales, respondiendo a la brecha de registro contextual identificada en trabajos previos.

1. Planificación de sesiones de captura: Se organizan las sesiones de captura considerando participantes voluntarios, superficies controladas, condiciones ambientales diferenciadas, disponibilidad del laboratorio y autonomía del equipo de adquisición. Esta planificación permite asegurar capturas completas, trazables y consistentes con el alcance experimental de cada fase del estudio.
2. Adquisición sincronizada de datos térmicos y ambientales: Se capturan secuencias térmicas posteriores al retiro de la mano y, en paralelo, se registran temperatura ambiente y humedad relativa mediante el instrumento ambiental definido. Estas variables no son manipuladas directamente, sino medidas como condiciones ambientales observadas. Su registro es relevante porque la literatura señala que las condiciones ambientales y de adquisición pueden modificar la lectura infrarroja y la evolución de la disipación térmica (Razmara et al., 2024; Holgado-Apaza et al., 2025; J.-Q. Li et al., 2024).
3. Construcción de la tabla maestra: Se consolida la información de cada prueba en una tabla estructurada que incluye rutas de imágenes térmicas, timestamps, participante, superficie, número de secuencia, tiempo transcurrido, temperatura ambiente y humedad relativa. En esta misma actividad se revisan archivos faltantes, nombres inconsistentes y registros incompletos. Esta organización permite mantener la trazabilidad de cada captura y facilita la asociación posterior entre imagen, contexto experimental y etiqueta temporal.
4. Estandarización inicial mediante ROI manual: Se delimita manualmente una región de interés para cada secuencia, procurando incluir la huella térmica completa y un margen de difusión alrededor del rastro. Este recorte inicial reduce regiones irrelevantes del fondo y concentra el análisis en la zona térmicamente significativa, sin aplicar todavía redimensionamiento, normalización ni transformación para modelos. Estas operaciones se realizarán en la etapa de preprocesamiento. La decisión se sustenta en estudios que recomiendan aislar la región térmica relevante y conservar información espacial antes del modelamiento convolucional o secuencial (Yu, Liang, Zhou, Zhang, et al., 2024; Movahedi-Rad & Keller, 2026; Garrido et al., 2021).

**Etapa 3. Preprocesamiento y generación de entradas para modelos**

En la actual etapa se transforman los datos del dataset base estandarizado en entradas específicas para las arquitecturas de estimación temporal. Esta etapa es necesaria porque, en imágenes térmicas, el preprocesamiento permite reducir ruido, aislar la región relevante, normalizar los datos radiométricos y representar la evolución temporal de la disipación. Además, permite integrar variables ambientales y experimentales, de modo que los modelos no dependan únicamente de la intensidad visual del rastro térmico (Holgado-Apaza et al., 2025; Ma et al., 2021; Lee et al., 2022; Razmara et al., 2024).

1. Preprocesamiento térmico de imágenes base: Se aplica sustracción de fondo cuando exista una referencia térmica de la superficie, filtrado suave si la imagen presenta ruido y normalización Min-Max o lineal de los valores térmicos. Estas técnicas permiten aislar la huella respecto al fondo y convertir los datos radiométricos en entradas adecuadas para redes convolucionales. Esta decisión se sustenta en estudios que emplean reducción de ruido, sustracción de fondo y normalización en imágenes térmicas (Ma et al., 2021; St-Antoine et al., 2025; Lee et al., 2022; Yu, Liang, Zhou & Zhang, 2024).
2. Preparación de variables ambientales y experimentales: Se integran la temperatura ambiente, la humedad relativa y el tipo de superficie como características asociadas a cada captura. Las variables numéricas serán estandarizadas para evitar diferencias de escala respecto a las entradas térmicas. Esta actividad se justifica porque la apariencia del rastro no depende únicamente del tiempo transcurrido, sino también de las condiciones de adquisición y del ambiente, las cuales pueden afectar la disipación y la lectura infrarroja (Razmara et al., 2024; Brezov et al., 2023; J.-Q. Li et al., 2024).
3. Generación de variables térmicas y temporales derivadas: Se calculan variables como temperatura media de la ROI, temperatura máxima, diferencia térmica respecto al ambiente, área térmica activa y ΔT entre capturas consecutivas. Estas variables permiten representar la intensidad, extensión y tasa de cambio de la huella térmica durante su disipación. El uso de diferencias térmicas entre cuadros permite capturar explícitamente la evolución temporal del calor, aspecto relevante para estimar el tiempo transcurrido (Lee et al., 2022; Xu et al., 2020).
4. Construcción de entradas específicas por arquitectura: Se generan las representaciones requeridas por cada modelo: imágenes normalizadas para DSTFS adaptado, secuencias térmicas para CNN-LSTM, imagen térmica junto con variables ambientales para CNN/ResNet multimodal, y curvas temporales de disipación para CNN 1D. Opcionalmente, se aplicará aumentación moderada solo en el conjunto de entrenamiento, mediante transformaciones leves que no alteren la coherencia física de la señal térmica (Yu, Liang, Zhou, Zhang, et al., 2024; Raj et al., 2025; Lee et al., 2022).

**Etapa 4. Implementación y entrenamiento de arquitecturas de estimación temporal**

En la actual etapa se implementan y entrenan arquitecturas de Deep Learning seleccionadas para estimar el tiempo transcurrido desde el retiro de la mano. Esta etapa se sustenta en la literatura revisada, donde se identifican modelos especializados en huellas térmicas, modelos secuenciales, enfoques multimodales y representaciones basadas en curvas de disipación. La selección prioriza arquitecturas compatibles con una fase experimental inicial, permitiendo evaluar distintas formas de representar el fenómeno: imagen térmica individual, secuencia temporal, integración ambiental y señal térmica reducida.

1. Implementación de DSTFS adaptado: Se implementa una arquitectura inspirada en DSTFS, debido a su aplicación directa en huellas térmicas de manos y estimación del tiempo de disipación. Este modelo resulta pertinente porque incorpora mecanismos de umbral suave para reducir ruido térmico e información redundante, lo cual es útil cuando la huella pierde contraste y nitidez con el tiempo (Yu, Liang, Zhou, Zhang, et al., 2024).
2. Implementación de CNN-LSTM: Se implementa una arquitectura CNN-LSTM para procesar secuencias térmicas de disipación. En este enfoque, una red convolucional extrae características espaciales de cada imagen térmica y una capa LSTM modela la evolución temporal de dichas características a lo largo de la secuencia. Su elección se justifica porque la huella térmica no debe analizarse solo como una imagen aislada, sino como una señal que cambia progresivamente con el tiempo. Además, frente a arquitecturas más complejas como ConvLSTM, CNN-LSTM resulta más adecuada para una fase experimental inicial, ya que permite representar la dinámica temporal con menor complejidad y menor riesgo de sobreajuste. Modelos recurrentes y espacio-temporales han mostrado utilidad en la predicción de fenómenos térmicos dinámicos (Boiko et al., 2022; Sajadi et al., 2024; Movahedi-Rad & Keller, 2026).
3. Implementación de CNN/ResNet multimodal: Se implementa un modelo con una rama convolucional para procesar la imagen térmica y una rama tabular para integrar temperatura ambiente, humedad relativa y tipo de superficie. Ambas representaciones se fusionan antes de la salida de regresión. Esta arquitectura se relaciona directamente con el aporte del estudio, ya que permite evaluar si las variables ambientales mejoran la estimación temporal del rastro térmico (Razmara et al., 2024; Brezov et al., 2023).
4. Implementación de CNN 1D sobre curvas de disipación: Se implementa una CNN 1D basada en curvas temporales extraídas de la región de interés, como temperatura media, temperatura máxima, diferencia térmica respecto al ambiente, ΔT entre capturas y área activa. Esta arquitectura permite representar la disipación como una señal compacta, reduciendo la dependencia de imágenes completas y evaluando si la tendencia térmica temporal es suficiente para estimar el tiempo transcurrido (Lee et al., 2022; Hao et al., 2023; Mentzel et al., 2021).

**Etapa 5. Validación preliminar y análisis de resultados iniciales**

En la actual etapa se realiza una validación preliminar de la propuesta, orientada a obtener resultados iniciales sobre el desempeño de los modelos de estimación temporal. Esta etapa no busca constituir una evaluación final, sino comprobar que el dataset construido, el flujo de preprocesamiento y las arquitecturas implementadas permiten generar predicciones cuantificables del tiempo transcurrido. Asimismo, permite identificar limitaciones iniciales y ajustes necesarios para fases posteriores del estudio.

1. Partición inicial del dataset: Se divide el dataset en subconjuntos de entrenamiento, validación y prueba preliminar, evitando que capturas de una misma secuencia se mezclen indebidamente entre particiones. Esta decisión busca reducir fuga de información, ya que imágenes consecutivas de una misma secuencia pueden presentar alta similitud temporal y espacial. De esta manera, la evaluación inicial refleja mejor la capacidad del modelo para generalizar a secuencias no vistas.
2. Validación funcional del pipeline: Se verifica que las imágenes térmicas, matrices radiométricas, variables ambientales, metadatos y entradas específicas puedan procesarse correctamente desde la captura hasta el entrenamiento inicial. Esta validación permite comprobar la consistencia del flujo metodológico, desde la adquisición sincronizada hasta la generación de entradas para cada arquitectura. También ayuda a detectar errores de formato, sincronización, normalización o asociación entre imágenes y registros ambientales.
3. Cálculo de métricas preliminares: Se reportan métricas iniciales de desempeño como MAE, RMSE y R², debido a que la estimación temporal se formula como una tarea de regresión. Estas métricas permiten medir el error promedio, penalizar desviaciones grandes y analizar la capacidad explicativa del modelo. Además, se considera Accuracy-60 o Accuracy-120 como métrica de tolerancia temporal, siguiendo enfoques aplicados en huellas térmicas de manos (Yu, Liang, Zhou & Zhang, 2024; Yu, Liang, Zhou, Zhang, et al., 2024).
4. Análisis preliminar de resultados: Se analizan tendencias iniciales por arquitectura, superficie, intervalo temporal y variables ambientales registradas. Este análisis permite identificar qué modelos presentan menor error, en qué condiciones la estimación resulta más difícil y si la temperatura ambiente o la humedad relativa aportan información útil. Los resultados obtenidos permitirán orientar ajustes posteriores en el protocolo, el preprocesamiento, las entradas de los modelos y la configuración de entrenamiento.

**EXPERIMENTACIÓN**

La presente sección describe el desarrollo experimental de la propuesta como una fase inicial de validación del pipeline metodológico. Se documentan las actividades realizadas para el diseño del protocolo, la adquisición de secuencias térmicas, el registro de variables ambientales, la construcción del dataset base, el preprocesamiento de datos y la preparación de entradas para los modelos. En esta fase, el estudio se orienta a comprobar la viabilidad del sistema de adquisición y del flujo de procesamiento con un conjunto inicial de participantes, dejando la ampliación del dataset para una fase posterior. Asimismo, se presentan evidencias de la experimentación con el fin de mostrar la trazabilidad entre la metodología planteada y su ejecución práctica.

**Etapa 1. Diseño experimental y sistema de adquisición**

En esta etapa se completó la definición del protocolo experimental y se implementó el sistema inicial de adquisición de datos térmicos. Primero, se estableció el procedimiento de captura para registrar rastros térmicos de la mano sobre superficies controladas, considerando participantes, superficies, duración de secuencia, tiempo de contacto, distancia de captura y registro ambiental. Esta definición permitió estandarizar el proceso antes de iniciar las sesiones principales de adquisición.

1. Definición del protocolo experimental

La Tabla 1 presenta el protocolo experimental definido para la adquisición inicial de rastros térmicos de mano. En esta fase se considera trabajar con 10 participantes voluntarios para validar el procedimiento de captura, el registro ambiental, la organización de secuencias y la construcción del dataset base. Se estandarizó el uso de la mano derecha, un tiempo de contacto de 10 segundos, una duración inicial de 10 minutos y una distancia cámara-superficie de aproximadamente 40 cm. La cámara fue ubicada en un montaje con inclinación aproximada de 60°, procurando mantener el paralelismo con la superficie evaluada. Además, se definió un esquema de captura variable, con mayor frecuencia durante el primer minuto por la rápida disipación inicial. En una fase posterior, el dataset será ampliado con 5 a 10 participantes adicionales y secuencias de mayor duración. Por ello, los resultados de esta fase serán interpretados como una validación preliminar del pipeline, no como una evaluación definitiva de generalización.

**Tabla 1**

*Protocolo experimental definido para la adquisición de rastros térmicos de mano*

| **Aspecto del protocolo** | **Definición establecida** | **Propósito dentro del estudio** |
| --- | --- | --- |
| Tipo de experimento | Captura controlada de rastros térmicos generados por contacto de la mano sobre superficies definidas | Garantizar que las imágenes térmicas provengan de un procedimiento estandarizado |
| Participantes | 10 participantes en fase inicial, ampliables a 15 o 20 en fase posterior | Obtener variabilidad inicial y fortalecer posteriormente la generalización del dataset |
| Superficies evaluadas | Madera y vidrio | Comparar la disipación térmica en materiales con propiedades distintas |
| Mano utilizada | Mano derecha para todos los participantes | Reducir la variabilidad asociada al uso de distintas manos |
| Tiempo de contacto | 10 segundos de contacto continuo entre la mano y la superficie | Controlar la cantidad inicial de calor transferido a la superficie |
| Forma de contacto | Apoyo directo de la mano sobre la superficie, evitando desplazamientos durante el contacto | Mantener una geometría inicial consistente del rastro térmico |
| Inicio de la captura | Inmediatamente después del retiro de la mano | Registrar la disipación desde el momento más cercano a la formación del rastro |
| Duración total de la secuencia | 10 minutos por prueba en la fase inicial, con posibilidad de ampliación en la fase posterior. | Registrar la disipación inicial e intermedia, y ampliar luego la observación de etapas tardías |
| Intervalo de captura | Cada 5 segundos durante el primer minuto, cada 10 segundos del minuto 1 al 5 y cada 30 segundos del minuto 5 al 10 | Registrar con mayor frecuencia los primeros instantes de disipación |
| Distancia cámara-superficie | Aproximadamente 40 cm | Mantener una escala visual relativamente constante entre capturas |
| Orientación de cámara | Montaje con inclinación aproximada de 60°, procurando paralelismo entre cámara y superficie | Reducir distorsiones geométricas y mejorar la consistencia de las tomas |
| Soporte experimental | Superficie y cámara ubicadas en un soporte replicable | Mantener una configuración física estable durante cada secuencia |
| Registro ambiental | Temperatura ambiente y humedad relativa durante cada medición | Asociar cada secuencia térmica con sus condiciones ambientales |
| Condiciones de ventilación | Tres escenarios: alto flujo de aire y baja temperatura, bajo flujo de aire y alta temperatura, y ausencia de ventilación | Evaluar la adquisición bajo condiciones ambientales diferenciadas |
| Variable objetivo | Tiempo transcurrido desde el retiro de la mano | Definir la etiqueta temporal que posteriormente será estimada por los modelos |
| Criterio de repetición | La prueba se repite si no se cuenta con una duración mínima suficiente de capturas | Asegurar una secuencia válida para analizar la disipación térmica |
| Criterio de exclusión | Se excluye toda medición sin registros ambientales de temperatura y humedad | Garantizar que cada secuencia tenga información térmica y ambiental asociada |

1. Selección de instrumentos de adquisición

La Tabla 2 resume los instrumentos seleccionados para la adquisición experimental. La cámara FLIR One Pro fue empleada para registrar la evolución térmica del rastro de mano, mientras que el Google Pixel 8 permitió ejecutar la aplicación móvil desarrollada en Android/Kotlin, controlar la captura secuencial y almacenar las imágenes generadas. Asimismo, el termohigrómetro Elitech RC-51H fue utilizado para registrar temperatura ambiente y humedad relativa durante cada medición. Finalmente, el soporte experimental permitió mantener una configuración física estable entre la cámara y la superficie evaluada.

**Tabla 2**

*Instrumentos seleccionados para la adquisición experimental*

| **Instrumento** | **Función en el experimento** | **Variable o dato registrado** | **Justificación de uso** |
| --- | --- | --- | --- |
| FLIR One Pro | Capturar imágenes térmicas del rastro residual de la mano | Imagen térmica de la huella durante la disipación | Permite registrar la evolución visual del calor residual sobre la superficie |
| Google Pixel 8 | Ejecutar la aplicación móvil de captura, conectarse con la cámara térmica y almacenar las imágenes generadas | Secuencias de imágenes, número de snapshot y timestamp | Permite automatizar la adquisición desde un dispositivo Android compatible y centralizar los archivos generados |
| Aplicación Android/Kotlin | Controlar la captura secuencial mediante la cámara térmica | Imágenes térmicas organizadas por secuencia | Reduce la captura manual y mantiene intervalos definidos entre tomas |
| Elitech RC-51H | Medir las condiciones ambientales durante cada prueba | Temperatura ambiente y humedad relativa | Permite asociar cada secuencia térmica con variables ambientales relevantes |
| Soporte experimental | Mantener una posición estable de la cámara y la superficie durante la captura | Distancia aproximada, inclinación del montaje y paralelismo cámara-superficie | Reduce variaciones de perspectiva, distancia y orientación entre pruebas |
| Superficies evaluadas | Recibir el contacto de la mano y conservar el rastro térmico | Rastro térmico generado sobre madera y vidrio | Permite analizar la disipación térmica en materiales definidos |

1. Desarrollo de la aplicación móvil Android/Kotlin

Como parte del sistema de adquisición, se desarrolló una aplicación móvil en Android/Kotlin para automatizar la captura de imágenes térmicas mediante la cámara FLIR One Pro conectada al Google Pixel 8. La Figura 2 muestra la interfaz inicial de la aplicación, desde la cual se gestiona la búsqueda de la cámara térmica y se visualiza el estado del sistema. Debido a que esta captura corresponde a una ejecución en emulador, su finalidad es evidenciar el diseño de la interfaz y el flujo inicial de conexión, no una adquisición térmica real.

La lógica principal de captura se evidencia en la Figura 3, donde se presenta el fragmento de código encargado de ejecutar una secuencia automática de 10 minutos. Esta función define un esquema de adquisición variable, con capturas cada 5 segundos durante el primer minuto, cada 10 segundos desde el minuto 1 hasta el minuto 5 y cada 30 segundos desde el minuto 5 hasta el minuto 10. Esta configuración permite registrar con mayor frecuencia la etapa inicial de disipación, donde se espera una variación térmica más rápida, y reducir la cantidad de capturas en los minutos posteriores.

Finalmente, las Figuras 4 y 5 evidencian la organización de los archivos generados por la aplicación. Cada prueba se almacena en una carpeta independiente identificada por fecha y hora de realización, mientras que cada imagen incluye un timestamp y un número de snapshot. Esta estructura facilita la trazabilidad de las secuencias, permite reconstruir el orden temporal de las capturas y sirve como base para asociar posteriormente cada imagen con el tiempo transcurrido, la superficie evaluada y las variables ambientales registradas.

**Figura 2**

*Interfaz inicial de conexión con la cámara FLIR*

![](data:image/png;base64...)

**Figura 3**

*Fragmento de código de la captura secuencial automática*

**![](data:image/png;base64...)**

**Figura 4**

*Carpetas generadas por sesión experimental*

![](data:image/png;base64...)

**Figura 5**

*Imágenes térmicas almacenadas con timestamp y número de snapshot*

**![](data:image/png;base64...)**

1. Pruebas piloto y ajuste operativo

Como parte de la validación inicial del sistema de adquisición, se realizaron pruebas piloto con el fin de verificar la visibilidad del rastro térmico y su evolución durante la disipación. La Figura 6 presenta una secuencia representativa obtenida en esta etapa, compuesta por capturas realizadas inmediatamente después del contacto de la mano con la superficie, así como a los 2, 5 y 10 minutos posteriores. Esta evidencia permitió confirmar que el rastro térmico presenta un mayor contraste en los primeros instantes y que su intensidad disminuye progresivamente con el paso del tiempo, hasta volverse menos distinguible respecto al fondo.

Estas pruebas también resultaron útiles para evaluar la viabilidad operativa del protocolo, ya que permitieron comprobar que la duración de 10 minutos y el esquema de captura definido eran suficientes para registrar distintas etapas de disipación. Asimismo, la observación de la pérdida progresiva de contraste reforzó la decisión de mantener una mayor frecuencia de captura durante el primer minuto, periodo en el que se espera una variación térmica más rápida. En conjunto, estas evidencias confirmaron que el montaje experimental y el sistema de adquisición eran funcionales para iniciar la construcción del dataset base.

**Figura 6**

*Secuencia piloto de disipación térmica de un rastro de mano sobre superficie controlada*

**![](data:image/jpeg;base64...)![](data:image/jpeg;base64...)![](data:image/jpeg;base64...)![](data:image/jpeg;base64...)**

**Etapa 2. Diseño experimental y sistema de adquisición**

En esta etapa se inició la construcción del dataset base a partir de las primeras sesiones de adquisición térmica. Para ello, se organizaron las jornadas de captura, se registraron secuencias de rastros térmicos bajo condiciones ambientales definidas y se comenzó la consolidación de los archivos generados en una estructura trazable. Asimismo, se inició la preparación inicial de las imágenes mediante la identificación de regiones de interés, con el propósito de estandarizar posteriormente las entradas para los modelos de estimación temporal.

1. Planificación de sesiones de captura

La planificación de las sesiones de captura se realizó considerando dos restricciones operativas principales: la reserva máxima de la sala de laboratorio, limitada a 3 horas, y la autonomía de la cámara FLIR One Pro, estimada entre 1 h 30 min y 2 h de uso continuo. Como se resume en la Tabla 3, estas condiciones llevaron a definir un número reducido de participantes por jornada, priorizando la obtención de secuencias completas y registros ambientales consistentes. En esta fase inicial se busca validar el procedimiento con 10 participantes, mientras que la ampliación del dataset se plantea para una fase posterior mediante la incorporación de nuevos participantes y secuencias de mayor duración. En cada sesión se registran variaciones ambientales definidas y la superficie evaluada se alterna entre madera y vidrio según la disponibilidad de los participantes y del espacio de trabajo.

**Tabla 3**

*Criterios operativos para la planificación de sesiones de captura*

| **Criterio** | **Definición establecida** | **Justificación** |
| --- | --- | --- |
| Duración máxima de reserva de sala | 3 horas por jornada | Limita el tiempo disponible para preparación, captura y cierre de sesión |
| Autonomía de cámara | Entre 1 h 30 min y 2 h de uso continuo | Restringe la cantidad de participantes medidos por día |
| Participantes por jornada | 2 participantes | Permite completar las secuencias sin exceder la autonomía del equipo |
| Superficie por jornada | Madera o vidrio, según disponibilidad | Facilita la organización flexible de sesiones |
| Variaciones ambientales | Sin ventilación, bajo flujo de aire con alta temperatura y alto flujo de aire con baja temperatura | Permite registrar la disipación bajo condiciones ambientales diferenciadas |
| Secuencias por jornada | 6 secuencias estimadas | Corresponde a 2 participantes evaluados bajo 3 condiciones ambientales |
| Orden de captura | Agrupado por participante y superficie | Reduce cambios de montaje y mejora la continuidad del procedimiento |
| Criterio de continuidad | Reprogramar capturas incompletas o sin registro ambiental | Evita incorporar secuencias no válidas al dataset |

1. Adquisición sincronizada de datos térmicos y ambientales

La Figura 7 muestra el montaje utilizado para la adquisición sincronizada de datos térmicos y ambientales. Se observa la disposición general del sistema, compuesto por el Google Pixel 8 conectado a la cámara FLIR One Pro, el soporte de captura, el termohigrómetro Elitech RC-51H y la superficie evaluada. Se presenta una vista lateral del montaje, donde se aprecia la orientación de la cámara respecto a la superficie. Esta configuración permitió registrar secuencias térmicas del rastro de mano y, de forma paralela, obtener mediciones de temperatura ambiente y humedad relativa durante cada prueba.

Estas evidencias permiten comprobar que la adquisición no se limitó al registro de imágenes térmicas, sino que incorporó variables ambientales asociadas a cada secuencia. De esta manera, cada medición puede vincularse posteriormente con su participante, superficie, condición ambiental, timestamp, tiempo transcurrido y valores de temperatura y humedad. Esta sincronización constituye un paso necesario para construir un dataset multimodal orientado a la estimación temporal de rastros térmicos.

**Figura 7**

*Montaje empleado para la adquisición sincronizada de datos térmicos y ambientales*

**![](data:image/jpeg;base64...)![](data:image/jpeg;base64...)**

1. Construcción de la tabla maestra

La construcción de la tabla maestra se encuentra en proceso, debido a que la recolección de datos térmicos y ambientales aún continúa. Como avance inicial, se desarrolló un pipeline en Python que organiza las imágenes crudas desde la carpeta raw\_data, extrae matrices térmicas de archivos FLIR, calcula el tiempo transcurrido dentro de cada secuencia y consolida los metadatos en archivos estructurados. Como se muestra en la Figura 8, los datos procesados se organizan por participante y superficie, y se generan archivos de salida como metadata y processing\_warnings, los cuales permiten centralizar la información útil y registrar advertencias del procesamiento.

La Figura 9 presenta una vista parcial de la tabla maestra generada. En esta se observa la asociación entre cada captura y sus principales metadatos, como identificador de muestra, secuencia, número de snapshot, participante, superficie, mano utilizada, rutas de archivos, fecha y hora de captura, tiempo transcurrido, temperatura ambiente, humedad relativa, método de asociación ambiental y dimensiones de la matriz térmica. Esta estructura permite mantener la trazabilidad entre las imágenes originales, los datos térmicos procesados y las variables ambientales registradas durante la adquisición.

No obstante, esta actividad aún se encuentra en depuración, ya que la adquisición de nuevas secuencias sigue en desarrollo. Por ello, algunos nombres de variables todavía serán ajustados para mantener una nomenclatura uniforme y compatible con las etapas posteriores de modelamiento. Asimismo, se está validando la integración de registros ambientales y la consistencia del procesamiento térmico, especialmente en casos donde una imagen no pueda ser procesada correctamente o donde se detecten inconsistencias en los metadatos. De esta manera, la tabla maestra funcionará como base estructurada para la posterior generación de entradas específicas para los modelos de estimación temporal.

**Figura 8**

*Organización inicial de datos procesados y archivos de control del pipeline en Python*

**![](data:image/png;base64...)**

**Figura 9**

*Vista parcial de la tabla maestra generada para la consolidación de metadatos térmicos y ambientales*

**![](data:image/png;base64...)**

1. Estandarización inicial mediante ROI manual

La estandarización inicial mediante ROI se encuentra en proceso, debido a que aún se están incorporando nuevas secuencias al dataset y se continúa evaluando la mejor estrategia para delimitar la zona térmicamente relevante. Como avance inicial, se implementó en Python un módulo de selección de región de interés que permite delimitar el área donde se ubica el rastro térmico de la mano. La Figura 10 muestra la selección manual de la ROI sobre una imagen piloto, donde se delimita la zona de análisis procurando incluir la huella completa y un margen alrededor de ella para conservar información asociada a la difusión térmica.

A partir de la región seleccionada, el pipeline genera imágenes recortadas y matrices térmicas asociadas para su posterior procesamiento. La Figura 11 presenta un ejemplo de imagen térmica recortada a partir de la ROI definida. En esta etapa no se aplica todavía redimensionamiento ni normalización, ya que estas operaciones serán realizadas posteriormente durante el preprocesamiento y la generación de entradas para los modelos.

Actualmente, esta actividad sigue en refinamiento. Se está evaluando una estrategia de selección automática o centrada de ROI con validación manual, con el objetivo de reducir el tiempo de procesamiento sin perder control sobre la calidad del recorte. Asimismo, se revisa que las regiones seleccionadas mantengan la huella visible durante la secuencia completa, especialmente en capturas posteriores donde el contraste térmico disminuye. Esta validación permitirá asegurar que las imágenes procesadas conserven información útil para la estimación temporal del rastro térmico.

**Figura 10**

*Selección manual de la región de interés sobre una imagen térmica piloto*

*![](data:image/png;base64...)*

**Figura 11**

*Imagen térmica recortada a partir de la región de interés seleccionada*

**![](data:image/jpeg;base64...)**

**REFERENCIAS**

Ai, J., Hu, M., Zhai, G., Zhang, X.-P., Wang, Y., Cai, L., Li, Q., & Sun, W. Q. (2020). Rapidly developing human heat residue model under various conditions based on Fluent and thermal video. Infrared Physics & Technology, 110, 103468. <https://doi.org/10.1016/j.infrared.2020.103468>

Bekhit, R., & Reimert, I. (2025). A Complete Pipeline to Extract Temperature from Thermal Images of Pigs. Sensors, 25(3), 643. <https://doi.org/10.3390/s25030643>

Boiko, D. A., Korabelnikova, V. A., Gordeev, E. G., & Ananikov, V. P. (2022). Integration of thermal imaging and neural networks for mechanical strength analysis and fracture prediction in 3D-printed plastic parts. Scientific Reports, 12(1), 8944. <https://doi.org/10.1038/s41598-022-12503-y>

Brezov, D., Hristov, H., Dimov, D., & Alexiev, K. (2023). Predicting the Rectal Temperature of Dairy Cows Using Infrared Thermography and Multimodal Machine Learning. Applied Sciences, 13(20), 11416. <https://doi.org/10.3390/app132011416>

Cho, K. W., Lin, F., Song, C., Xu, X., Gu, F., & Xu, W. (2016). Thermal handprint analysis for forensic identification using Heat-Earth Mover’s Distance. 2016 IEEE International Conference on Identity, Security and Behavior Analysis (ISBA), 1–8. <https://doi.org/10.1109/ISBA.2016.7477241>

Contreras, K., Toscano-Palomino, L., Mura, M. D., & Bacca, J. (2025). See the past: Time-Reversed Scene Reconstruction from Thermal Traces Using Visual Language Models (arXiv:2510.05408). arXiv. <https://doi.org/10.48550/arXiv.2510.05408>

Dar, F., Emenike, H., Yin, Z., Liyanage, M., Sharma, R., Zuniga, A., Hoque, M. A., Radeta, M., Nurmi, P., & Flores, H. (2022). The MIDAS touch: Thermal dissipation resulting from everyday interactions as a sensing modality. Pervasive and Mobile Computing, 84, 101625. <https://doi.org/10.1016/j.pmcj.2022.101625>

Elmessery, W. M., Habib, A., Shams, M. Y., Abd El-Hafeez, T., El-Messery, T. M., Elsayed, S., Fodah, A. E. M., Abdelwahab, T. A. M., Ali, K. A. M., Osman, Y. K. O. T., Abdelshafie, M. F., El-wahhab, G. G. A., & Elwakeel, A. E. (2024). Deep regression analysis for enhanced thermal control in photovoltaic energy systems. Scientific Reports, 14(1), 30600. <https://doi.org/10.1038/s41598-024-81101-x>

Esposito, M., Sessa, F., Cocimano, G., Zuccarello, P., Roccuzzo, S., & Salerno, M. (2023). Advances in technologies in crime scene investigation. Diagnostics, 13(20), 3169. <https://doi.org/10.3390/diagnostics13203169>

Garrido, I., Barreira, E., Almeida, R. M. S. F., & Lagüela, S. (2021). Building Façade Protection Using Spatial and Temporal Deep Learning Models applied to Thermographic Data. Laboratory Tests. The 16th International Workshop on Advanced Infrared Technology & Applications, 20. <https://doi.org/10.3390/engproc2021008020>

Golosov, N., & Cervone, G. (2024). Integrating Thermal Infrared Imaging and Weather Data for Short-Term Prediction of Building Envelope Thermal Appearance. Remote Sensing, 16(21), 3981. <https://doi.org/10.3390/rs16213981>

Hao, L., Li, Q., Pan, W., Yao, R., & Liu, S. (2023). Ice accretion thickness prediction using flash infrared thermal imaging and BP neural networks. IET Image Processing, 17(3), 649–659. <https://doi.org/10.1049/ipr2.12662>

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (arXiv:1512.03385). arXiv. <https://doi.org/10.48550/arXiv.1512.03385>

Holgado-Apaza, L. A., Prieto-Luna, J. C., Carpio-Vargas, E. E., Ulloa-Gallardo, N. J., Vilchez-Navarro, Y., Barrón-Adame, J. M., Aguirre-Puente, J. A., Ramos Enciso, D., Castellon-Apaza, D. D., & Saman-Pacamia, D. J. (2025). PropNet-R: A Custom CNN Architecture for Quantitative Estimation of Propane Gas Concentration Based on Thermal Images for Sustainable Safety Monitoring. Sustainability, 17(21), 9801. <https://doi.org/10.3390/su17219801>

Hou, F., Zhang, Y., Zhou, Y., Zhang, M., Lv, B., & Wu, J. (2022). Review on infrared imaging technology. Sustainability, 14(18), 11161. <https://doi.org/10.3390/su141811161>

Instituto Nacional de Estadística e Informática. (2023). Homicidios en el Perú, contándolos uno a uno, 2021 [Informe]. INEI. <https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1927/libro.pdf>

J, N. T., & Thinakaran, K. (2023). Detection of Crime Scene Objects using Deep Learning Techniques. 2023 International Conference on Intelligent Data Communication Technologies and Internet of Things (IDCIoT), 357–361. <https://doi.org/10.1109/IDCIoT56793.2023.10053440>

Kaczmarek, T., Ozturk, E., & Tsudik, G. (2018). Thermanator: Thermal Residue-Based Post Factum Attacks On Keyboard Password Entry (arXiv:1806.10189). arXiv. <https://doi.org/10.48550/arXiv.1806.10189>

Kim, H., & Hong, T. (2024). Enhancing emotion recognition using multimodal fusion of physiological, environmental, personal data. Expert Systems with Applications, 249, 123723. <https://doi.org/10.1016/j.eswa.2024.123723>

Lee, D.-G., Song, K.-S., Nho, Y.-H., Kim, A., & Kwon, D.-S. (2022). Sequential thermal image-based adult and baby detection robust to thermal residual heat marks. 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 13120–13127. <https://doi.org/10.1109/IROS47612.2022.9982115>

Li, H., Wen, S., Li, S., Wang, H., Geng, X., Wang, S., Zhai, J., & Zhang, W. (2024). The research on infrared radiation affected by smoke or fog in different environmental temperatures. Scientific Reports, 14(1), 14410. <https://doi.org/10.1038/s41598-024-65462-x>

Li, J.-Q., Xia, X.-L., Sun, C., & Chen, X. (2024). Estimation of time-dependent laser heat flux distribution based on BPNN improved by multiple population genetic algorithm. International Journal of Heat and Mass Transfer, 233, 125997. <https://doi.org/10.1016/j.ijheatmasstransfer.2024.125997>

Li, Y., Mo, F., Tang, F., Sun, B., & Zhou, C. (2024). A temperature measurement compensation method for industrial rotary kilns based on infrared multi-feature fusion under dynamic water mist interference. Infrared Physics & Technology, 141, 105485. <https://doi.org/10.1016/j.infrared.2024.105485>

Lilhore, U. K., Simaiya, S., Singh, R. K., Baqasah, A. M., Alroobaea, R., Alsafyani, M., Alhazmi, A., & Khan, M. D. M. (2025). Advanced air quality prediction using multimodal data and dynamic modeling techniques. Scientific Reports, 15(1), 27867. <https://doi.org/10.1038/s41598-025-11039-1>

Lloyd Institute of Forensic Science. (n.d.). How crime scenes are reconstructed using forensics [Blog post]. LIFS. <https://lifs.co.in/blog/crime-scene-reconstruction-using-forensics.html>

Luo, Q., Gao, B., Woo, W. L., & Yang, Y. (2019). Temporal and spatial deep learning network for infrared thermal defect detection. NDT & E International, 108, 102164. <https://doi.org/10.1016/j.ndteint.2019.102164>

Ma, G., Ross, W., Tucker, M., Hsu, P.-C., Buckland, D. M., & Codd, P. J. (2021). Touch-Point Detection Using Thermal Video With Applications to Prevent Indirect Virus Spread. IEEE Journal of Translational Engineering in Health and Medicine, 9, 1–11. <https://doi.org/10.1109/JTEHM.2021.3083098>

Maino, A., Alberi, M., Barbagli, A., Chiarelli, E., Colonna, T., Franceschi, M., Gallorini, F., Guastaldi, E., Lopane, N., Mantovani, F., Petrone, D., Pierini, S., Raptis, K. G. C., Strati, V., & Xhixha, G. (2024). A deep neural network for predicting soil texture using airborne radiometric data. Radiation Physics and Chemistry, 221, 111767. <https://doi.org/10.1016/j.radphyschem.2024.111767>

Mazdeyasna, S., Ghassemi, P., & Wang, Q. (2023). Best Practices for Body Temperature Measurement with Infrared Thermography: External Factors Affecting Accuracy. Sensors, 23(18), 8011. <https://doi.org/10.3390/s23188011>

Mentzel, F., Derugin, E., Jansen, H., Kröninger, K., Nackenhorst, O., Walbersloh, J., & Weingarten, J. (2021). No more glowing in the dark: How deep learning improves exposure date estimation in thermoluminescence dosimetry. Journal of Radiological Protection, 41(4), S506–S521. <https://doi.org/10.1088/1361-6498/ac20ae>

Movahedi-Rad, A. V., & Keller, T. (2026). Load history effects in fiber–polymer composites: A CRNN-based hybrid deep learning approach for fatigue life prediction and structural health monitoring via infrared thermography. Composites Part A: Applied Science and Manufacturing, 200, 109263. <https://doi.org/10.1016/j.compositesa.2025.109263>

Nikolov, I., Philipsen, M. P., Liu, J., Dueholm, J. V., Johansen, A. S., Nasrollahi, K., & Moeslund, T. B. (2021). Seasons in Drift: A Long-Term Thermal Imaging Dataset for Studying Concept Drift. <https://openreview.net/forum?id=LjjqegBNtPi>

Raj, S., Pradhan, S. R., Anand, P., Deswal, D., Shah, S. S. H., & Kumar, V. (2025). Deep Learning-Based Object Detection in Thermal Imaging. 2025 Global Conference in Emerging Technology (GINOTECH), 1–6. <https://doi.org/10.1109/GINOTECH63460.2025.11076813>

Razmara, P., Khezresmaeilzadeh, T., & Jenkins, B. K. (2024). Fever Detection with Infrared Thermography: Enhancing Accuracy through Machine Learning Techniques. 2024 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI), 1–8. <https://doi.org/10.1109/BHI62660.2024.10913591>

Sachan, R., Kundra, S., & Dubey, A. K. (2022). An Efficient Algorithm for Object Detection in Thermal Images using Convolutional Neural Networks and Thermal Signature of the Objects. 2022 4th International Conference on Energy, Power and Environment (ICEPE), 1–6. <https://doi.org/10.1109/ICEPE55035.2022.9798144>

Sajadi, P., Dehaghani, M. R., Tang, Y., & Wang, G. G. (2024). Real-Time 2D Temperature Field Prediction in Metal Additive Manufacturing Using Physics-Informed Neural Networks. <https://doi.org/10.48550/arXiv.2401.02403>

Senthilraj, S., & Shanker, N. R. (2023). A Condition Monitoring System for Electric Vehicle Batteries Based on a Convolutional Neural Network Using Thermal Image. International Journal on Recent and Innovation Trends in Computing and Communication, 11(6), 52–62. <https://doi.org/10.17762/ijritcc.v11i6.6772>

Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W., & Woo, W. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (arXiv:1506.04214). arXiv. <https://doi.org/10.48550/arXiv.1506.04214>

St-Antoine, C., Paradis, M.-C. M., Gonnel, P., Foran, G., Therrien, F., Prébé, A., & Dollé, M. (2025). Novel method to automatize flash point detection in small volumes of liquid by computer vision using thermal images. Measurement, 253, 117629. <https://doi.org/10.1016/j.measurement.2025.117629>

Strąkowska, M., & Strzelecki, M. (2023). Thermal Time Constant CNN-Based Spectrometry for Biomedical Applications. Sensors, 23(15), 6658. <https://doi.org/10.3390/s23156658>

Tang, B., & Sato, W. (2026). Machine Learning-Based Ear Thermal Imaging for Emotion Sensing. Sensors, 26(4), 1248. <https://doi.org/10.3390/s26041248>

Tang, B., Sato, W., & Kawanishi, Y. (2025). Development of Machine-Learning-Based Facial Thermal Image Analysis for Dynamic Emotion Sensing. Sensors, 25(17), 5276. <https://doi.org/10.3390/s25175276>

Tang, Z., Ye, W., Ma, W.-C., & Zhao, H. (2023). What Happened 3 Seconds Ago? Inferring the Past with Thermal Imaging. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17111–17120. <https://doi.org/10.1109/CVPR52729.2023.01641>

United Nations Office on Drugs and Crime. (2023). Global study on homicide 2023 (4th ed.). United Nations. <https://www.unodc.org/documents/data-and-analysis/gsh/2023/Global_study_on_homicide_2023_web.pdf>

Usamentiaga, R., Venegas, P., Guerediaga, J., Vega, L., Molleda, J., & Bulnes, F. (2014). Infrared Thermography for Temperature Measurement and Non-Destructive Testing. Sensors, 14(7), 12305–12348. <https://doi.org/10.3390/s140712305>

Wickenheiser, R. A. (2023). Proactive crime scene response optimizes crime investigation. Forensic Science International: Synergy, 6, 100325. <https://doi.org/10.1016/j.fsisyn.2023.100325>

Wilk, L. S., Edelman, G. J., Roos, M., Clerkx, M., Dijkman, I., Melgar, J. V., Oostra, R.-J., & Aalders, M. C. G. (2021). Individualised and non-contact post-mortem interval determination of human bodies using visible and thermal 3D imaging. Nature Communications, 12(1), 5997. <https://doi.org/10.1038/s41467-021-26318-4>

Xiao, P., & Chen, D. (2024). Photothermal Radiometry Data Analysis by Using Machine Learning. Sensors, 24(10), 3015. <https://doi.org/10.3390/s24103015>

Xu, Z., Wang, Q., Li, D., Hu, M., Yao, N., & Zhai, G. (2020). Estimating Departure Time Using Thermal Camera and Heat Traces Tracking Technique. Sensors, 20(3), 782. <https://doi.org/10.3390/s20030782>

Yu, X., Liang, X., Zhou, Z., & Zhang, B. (2024). Multi-task learning for hand heat trace time estimation and identity recognition. Expert Systems with Applications, 255, 124551. <https://doi.org/10.1016/j.eswa.2024.124551>

Yu, X., Liang, X., Zhou, Z., Zhang, B., & Xue, H. (2024). Deep soft threshold feature separation network for infrared handprint identity recognition and time estimation. Infrared Physics & Technology, 138, 105223. <https://doi.org/10.1016/j.infrared.2024.105223>

Yu, X., Ye, X., & Gao, Q. (2020). Infrared Handprint Image Restoration Algorithm Based on Apoptotic Mechanism. IEEE Access, 8, 47334–47343. <https://doi.org/10.1109/ACCESS.2020.2979018>

Zhao, S., Liu, Y., Jiao, Q., Zhang, Q., & Han, J. (2024). Mitigating Modality Discrepancies for RGB-T Semantic Segmentation. IEEE Transactions on Neural Networks and Learning Systems, 35(7), 9380–9394. <https://doi.org/10.1109/TNNLS.2022.3233089>

Zhou, Z., Zhang, B., & Yu, X. (2021). Infrared Handprint Classification Using Deep Convolution Neural Network. Neural Processing Letters, 53(2), 1065–1079. <https://doi.org/10.1007/s11063-021-10429-6>

Zhou, Z., Zhang, B., & Yu, X. (2022). Immune coordination deep network for hand heat trace extraction. Infrared Physics & Technology, 127, 104400. <https://doi.org/10.1016/j.infrared.2022.104400>