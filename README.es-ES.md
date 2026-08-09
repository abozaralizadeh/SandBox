

# SandBox

Estos proyectos no son herramientas aisladas, sino facetas de una exploración más amplia sobre la autonomía de la IA. La presencia de estos tres proyectos distintos pero relacionados revela una visión cohesiva. Mi trabajo no se trata simplemente de código, sino de explorar el potencial de la IA agente para interactuar y razonar sobre el mundo real de formas cada vez más sofisticadas. Los temas asociados del repositorio, como la gobernanza de la IA, la ética y la simulación, refuerzan este enfoque en las implicaciones de alto orden de los sistemas autónomos.

## ComicBook

Generador de Historietas con IA Multiagente  
ComicBook produce una historieta diaria generada por IA en el **OpenAI Agents SDK**, orquestada como una **cadena de handoff (relevo)**: Director → Storyteller → Cartoonist → Reteller. Una sola llamada a `Runner.run(Director)` impulsa todo el episodio; cada agente invoca su herramienta `transfer_to_<next>` para pasar el control, y una **recuperación determinista** ejecuta cualquier etapa que un handoff fallido haya omitido, asegurando que la historieta se publique siempre. El Director también consulta a un subagente **OriginalityCritic** (vía `as_tool`) antes de iniciar un nuevo arco narrativo.
[Read more](https://medium.com/towards-artificial-intelligence/the-comic-that-draws-itself-building-a-daily-ai-graphic-novel-studio-da4f7a61e39c)

📐 **Architecture:** [Technical flow and Mermaid diagrams](ComicBook/architecture.md)

#### Agent Pipeline
- **Director** (temp 1.2) — Inventa arcos narrativos originales. Para un nuevo arco, **busca en la web inspiración fresca**, forma un candidato, llama a `check_arc_originality` y reintenta hasta que sea distinto de los arcos recientes; decide la longitud del arco de forma orgánica; escribe el esquema de la historia; planifica los paneles y el tono de cada episodio. Luego pasa el control al Storyteller.
- **OriginalityCritic** (temp 0.2, `as_tool`) — Lee los arcos recientes y evalúa la historia central de un candidato (estructura de la trama, conflicto, arquetipos, entorno, estilo artístico), devolviendo `ok` / `too_similar` + orientación para reintento. `start_new_arc` además rechaza un estilo artístico usado recientemente.
- **Storyteller** (temp 0.5) — Transforma el plan del Director en un guion panel por panel (diálogos, subtítulos, SFX, ángulos de cámara, tamaño por panel) y luego pasa el control al Cartoonist.
- **Cartoonist** — Obtiene el listado completo del arco, genera una **ficha de referencia de personajes** para consistencia visual, luego dibuja cada panel secuencialmente usando la referencia mediante edición de imágenes de Azure OpenAI, ensambla la página HTML en inglés y pasa el control al Reteller.
- **Reteller** (temp 0.9) — En una sola ejecución, **reinterpreta** el episodio nativamente en italiano y persa sobre las mismas imágenes fijas (no es una traducción), adaptando y guardando el esquema localizado en el episodio 1 y manteniendo un glosario por idioma.

#### Key Features
- **Dynamic Story Arcs**: los arcos duran tantos episodios como necesiten (3, 8, 15…), luego se cierran y comienza un arco completamente nuevo. Una **protección de originalidad en tres capas** (búsqueda obligada por el prompt + el OriginalityCritic + un rechazo de estilo artístico) mantiene cada arco genuinamente distinto.
- **Character Consistency**: una ficha de referencia de personajes por arco (en caché), luego cada panel se dibuja secuencialmente con esa referencia (más paneles clave a mitad de arco y anclas de episodios anteriores).
- **Multi-language editions**: el inglés es nativo; el Reteller produce versiones en it/fa sobre el arte compartido. El **título principal proviene del arco** (consistente en cada episodio) y el título nativo del episodio se muestra como **subtítulo**.
- **Readability guard**: el tema de colores de la página se verifica en contraste al renderizar — cualquier texto de bajo contraste se invierte automáticamente a casi negro/casi blanco, para que un recuadro nunca sea claro-sobre-claro u oscuro-sobre-oscuro.
- **Local debug mode**: `DEBUG=true` aísla todas las lecturas/escrituras en una partición `arc_debug` separada (y bloqueo) para que las pruebas locales nunca toquen la producción; `DEBUG_SAVE=false` es una ejecución en seco pura.
- **Arc Memory**: Azure Table Storage rastrea metadatos de arco, esquemas, glosarios, paneles clave y resúmenes de episodio para que cada tira respete la continuidad.
- **Responsive Comic Layout**: estilo de cómic (bocadillos de diálogo, recuadros de subtítulo, SFX) que se adapta a móvil y escritorio.
- **Frontend**: `/comicbook` obtiene la tira más reciente (o una fecha seleccionada) y renderiza la página del cómic.

#### Tech Stack
- **OpenAI Agents SDK** (`openai-agents`) — definiciones de agentes, herramientas `@function_tool` (solo deterministas), **handoffs** con `input_filter` + `prompt_with_handoff_instructions`, subagentes vía `Agent.as_tool`, `Runner.run()`
- **Azure OpenAI** — modelo de chat (configurable, ej. `gpt-5.4`) para los agentes, `gpt-image` para generación de imágenes (tiempos de espera de cliente de 1 hora, anular vía `COMICBOOK_LLM_TIMEOUT` / `COMICBOOK_IMAGE_TIMEOUT`)
- **Azure Table & Blob Storage** — persistencia de arco/episodio (con aislamiento de partición `DEBUG`), alojamiento de imágenes y HTML

![ComicBook](https://github.com/abozaralizadeh/SandBox/blob/main/static/ComicBook.png?raw=true)

## AI Open Problem Solver

Matemático Creativo Autónomo y Agente de Investigación Profunda  
AI Open Problem Solver extiende la infraestructura de LangGraph con un matemático autónomo creativo que intenta activamente resolver problemas matemáticos abiertos, incluidos los Problemas del Milenio. En lugar de simplemente buscar en internet trabajos existentes, el agente formula conjeturas originales, ejecuta experimentos computacionales, desarrolla estrategias de demostración y utiliza la investigación web solo como complemento. Cada iteración diaria se registra como una entrada rica en formato HTML de cuaderno de laboratorio. [Read more](https://pub.towardsai.net/can-my-autonomous-ai-agent-solve-a-millennium-problem-and-win-1-000-000-3ff8fff8a786)

📐 **Architecture:** [Technical flow and Mermaid diagrams](AIOpenProblemSolver/architecture.md)

#### Key Capabilities
- **Creative Problem Solving**: El agente piensa de forma independiente: formula hipótesis, prueba conjeturas computacionalmente, construye bocetos de demostración y cambia de rumbo cuando los enfoques fallan, en lugar de resumir pasivamente investigaciones existentes.
- **Python Math Sandbox**: Una herramienta computacional dedicada le brinda al agente un laboratorio matemático completo con SymPy (álgebra/cálculo simbólico), NumPy (cálculo numérico), SciPy (optimización, integración, funciones especiales) y Matplotlib, lo que le permite ejecutar experimentos, verificar demostraciones numéricamente y buscar contraejemplos.
- **Symbolic Calculator**: Una herramienta ligera para operaciones simbólicas rápidas (simplificar, factorizar, resolver, integrar, diferenciar, expansión en series) sin escribir scripts completos de Python.
- **Deep Search Agent**: Utiliza el flujo de trabajo de búsqueda profunda de LangGraph (con un fallback ReAct) para complementar su propio razonamiento con investigación web: consultando teoremas específicos, verificando si se ha probado un enfoque o encontrando artículos relevantes.
- **Tuned for Creativity**: Una temperatura de LLM más alta (configurable vía `AIOPS_LLM_TEMPERATURE`) fomenta un pensamiento matemático novel y diverso a lo largo de las iteraciones.
- **Persistent Research Memory**: Almacena cada actualización diaria, junto con resúmenes estructurados, siguientes pasos y citas en Azure Table Storage.
- **Resume & Continue**: En cada ejecución, el agente ingiere el contexto histórico y avanza en el mismo problema en lugar de comenzar desde cero.
- **Infinite Timeline UI**: Una ruta de Flask (`/ai-open-problem-solver`) sirve una página de desplazamiento infinito que transmite primero los hallazgos más recientes y carga perezosamente hitos anteriores desde el almacenamiento.
- **Dynamic Problem Picker**: La interfaz consulta Azure Table Storage para poblar un menú desplegable con todos los problemas rastreados, para que puedas cambiar hilos de investigación instantáneamente.
- **Problem Catalog Table**: Configura `aiops_problem_table_name` para la tabla dedicada del registro de problemas; agrega o elimina problemas allí para controlar qué hilos están disponibles en la interfaz.

Configura los nombres de almacenamiento (`aiops_table_name`, `aiops_blob_name`) y opcionalmente define `AIOPS_DEFAULT_PROBLEM` en `.env` para elegir el problema sin resolver predeterminado que abordará el agente. Ajusta la creatividad con `AIOPS_LLM_TEMPERATURE` (valor predeterminado `0.8`) y el tiempo de ejecución del sandbox con `AIOPS_SANDBOX_TIMEOUT` (valor predeterminado `120` segundos).

Live Demo
View the daily progress of AI open problems solver:
https://sandboxes.live/ai-open-problem-solver

![AIBlog](https://raw.githubusercontent.com/abozaralizadeh/SandBox/refs/heads/main/static/AiProblemSolverImage.png)

## AI Blog

Un Investigador de IA Autónomo que Escribe sobre IA
AIBlog es un agente autónomo que descubre nuevos desarrollos en aprendizaje automático e IA generativa, realiza investigación multisource y publica un artículo de blog diario escrito completamente por IA. Utiliza un agente ReAct envuelto en una máquina de estados de LangGraph para orquestar búsquedas web, lectura académica, síntesis y composición final en HTML limpio.

#### Each post is automatically:

- Researched via arXiv, OpenAI, DeepMind, HuggingFace, and other credible sources.
- Enriched with citations, code snippets, tables, and technical diagrams.
- Published as a fully responsive, styled HTML article.
- Illustrated with a custom banner image generated by DALL·E 3 and hosted on Azure Blob Storage.

AIBlog muestra el potencial de las arquitecturas de agentes recursivos para crear contenido técnico de alta calidad y verificable sin intervención humana. [Read more](https://medium.com/design-bootcamp/recursive-intelligence-an-ai-agent-that-researches-and-writes-about-ai-autonomously-100bccd81001)

📐 **Architecture:** [Technical flow and Mermaid diagrams](AIBlog/architecture.md)

Live Demo
View the daily AI-written blog:
https://SandBoxes.Live/aiblog

![AIBlog](https://raw.githubusercontent.com/abozaralizadeh/SandBox/refs/heads/main/static/AIBlog.png)

## Tomorrow News

Predicción de Noticias y Toma de Decisiones Impulsadas por IA
TomorrowNews es un proyecto experimental de código abierto que utiliza LangChain Agents y Azure OpenAI para generar predicciones de noticias especulativas impulsadas por IA basadas en eventos del mundo real. El proyecto tiene como objetivo simular la toma de decisiones para el futuro, ofreciendo un vistazo creativo a lo que podría suceder en diversos sectores, como la política, la economía, la sociedad y el medio ambiente.

Al alimentar con noticias reales como entrada, este proyecto genera predicciones y resultados para el día siguiente, creando titulares especulativos y decisiones relacionadas con temas globales. [Read more](https://medium.com/@abozar-alizadeh/tomorrow-news-speaks-three-languages-how-ai-reports-the-future-in-multiple-languages-473f2303b284)

📐 **Architecture:** [Technical flow and Mermaid diagrams](TomorrowNews/architecture.md)

#### Key Features:
- **Autonomous AI Predictions**: Utiliza datos de noticias en tiempo real para predecir eventos futuros plausibles.
- **Generative AI Agents**: Impulsado por Azure OpenAI, el sistema crea artículos de periódico detallados e imaginativos.
- **Responsive HTML Layout**: El resultado final es una página de periódico diseñada con elegancia, optimizada para pantallas de escritorio y móviles.
- **Dynamic Image Generation**: Incorpora imágenes generadas por IA que complementan los titulares, asegurando una experiencia visual cohesiva y atractiva.

### Core Components

#### 1. **LangGraph Framework**
LangGraph es la columna vertebral de este proyecto, proporcionando un entorno con estado y multiactor para construir flujos de trabajo de agentes. Las características clave de LangGraph incluyen:

- **Cycles and Branching**: Permite la implementación de bucles y condicionales dentro de la aplicación.
- **Persistence**: Guarda el estado de la aplicación después de cada paso, apoyando la recuperación de errores y flujos de trabajo con intervención humana.
- **Human-in-the-Loop**: Habilita la interrupción de la ejecución del grafo para aprobación o ediciones humanas.
- **Streaming Support**: Las salidas se transmiten a medida que son generadas por cada nodo.
- **Integration with LangChain**: Se integra sin problemas con LangChain y LangSmith para mejorar la funcionalidad.

#### 2. **Azure OpenAI Integration**
El proyecto aprovecha Azure OpenAI para generar contenido de noticias e imágenes. Usando GPT-4, los modelos de IA analizan los eventos actuales y generan predicciones para el periódico del día siguiente.

#### 3. **Tools and Agents**
- **News Feed Tool**: Obtiene las últimas noticias para proporcionar a la IA el contexto necesario para las predicciones.
- **Image Generation Tool**: Crea imágenes realistas basadas en prompts detallados para mejorar el atractivo visual del periódico.
- **Agent Workflow**: El agente procesa el feed de noticias, genera predicciones y formatea la salida en una página HTML. Este proceso implica múltiples iteraciones y pasos de toma de decisiones para garantizar contenido de alta calidad.

### How It Works

1. **News Collection**: El sistema obtiene las últimas noticias cada hora utilizando la News Feed Tool.
2. **AI Analysis and Prediction**: El agente de IA generativa analiza las noticias actuales y predice eventos futuros potenciales.
3. **Content Creation**: El agente crea artículos detallados y genera imágenes adecuadas utilizando la Image Generation Tool.
4. **HTML Newspaper Generation**: El contenido se formatea en una página HTML que se asemeja a un diseño de periódico tradicional, completa con titulares, artículos e imágenes.
5. **Output Delivery**: La página HTML está lista para ser renderizada en un navegador, ofreciendo a los usuarios una visión especulativa de las noticias del mañana.


Note: All content generated by this project is purely speculative, based on AI's interpretation of current events, and should not be viewed as factual or actual news predictions.

Live Demo
You can view the live version of the project here:
https://SandBoxes.Live/tomorrownews

![TN](https://github.com/abozaralizadeh/SandBox/blob/main/static/TomorrowNewsSample4.png?raw=true)

## GenBox

Este proyecto experimental explora el potencial de la IA como tomador de decisiones autónomo para un mundo virtual. Utilizando Azure OpenAI y un bucle estructurado de prompt-respuesta, el sistema genera decisiones de alto nivel diarias sobre áreas críticas como economía, sociedad, medio ambiente y política global. Cada decisión está diseñada para ser realista, impactante y éticamente informada, equilibrando resultados inmediatos con sostenibilidad a largo plazo. El objetivo es crear una narrativa atractiva y evolutiva que demuestre las capacidades de la IA generativa, al tiempo que invita a los usuarios a reflexionar sobre la gobernanza y las complejidades de la toma de decisiones en un mundo simulado. [Read more](https://medium.com/@abozar-alizadeh/giving-the-ai-government-a-face-and-a-voice-building-genboxs-self-producing-newsroom-db7f57b9e1c7)

**Grounded in the real world:** cada decisión se construye en dos fases. Primero, la IA **selecciona un tema fresco** para el día (diversificado lejos de los recientes). Una vez establecido el tema — pero *antes* de escribir la decisión — GenBox ejecuta una fase de investigación en internet utilizando la **búsqueda web nativa** del modelo (la herramienta `web_search` de la API Azure OpenAI Responses, el mismo enfoque que usan AIBlog y ComicBook), recoplando los **logros, desafíos, bloqueos y límites** del mundo real actual sobre ese tema. Ese informe se alimenta a la segunda fase, por lo que la decisión de la IA propone soluciones concretas y aplicables a problemas actuales *reales* en lugar de políticas abstractas; el tema elegido y las URL de origen en las que se basó se guardan junto con la decisión. La investigación es de "mejor esfuerzo": si la búsqueda web no está disponible, la decisión aún se produce (solo sin anclaje).

Las decisiones se presentan en una televisión CRT retro: la experiencia clásica hace desplazamiento de la decisión del día como texto verde de terminal, y los cuatro botones físicos te permiten avanzar/retroceder a través de la línea de tiempo y pausar o revertir el desplazamiento.

📐 **Architecture:** [Technical flow and Mermaid diagrams](GenBox/architecture.md)

Te invito a explorar la interfaz muy sencilla en https://SandBoxes.Live/genbox, donde puedes presenciar las decisiones diarias de la IA y seguir la narrativa evolutiva de este mundo virtual.

![TV](https://github.com/abozaralizadeh/SandBox/blob/main/static/sample.png?raw=true)

### AI News Anchor (Sora 2 Video)

GenBox ahora transmite cada decisión diaria como un **segmento de noticias de TV** generado por IA que se reproduce en la CRT en lugar del texto con desplazamiento. En lugar de leer la decisión (a menudo larga) textualmente, se cubre como un boletín real: un titular corto del presentador ("The AI Government today decided to…"), un informe de campo con material de cobertura (b-roll) y una breve entrevista, todo con voz y sincronización labial generadas nativamente por **Sora 2 en Azure OpenAI**. Cuando hay un video disponible, se convierte en el modo de visualización predeterminado; el texto de desplazamiento original siempre está a una pulsación de botón de distancia.

#### Key Features
- **News-Bulletin Format**: Un agente "Producer" del **OpenAI Agents SDK** destila la decisión (no una lectura textual) en una lista de planos segmentada — **anchor lead → field report → interview → sign-off** — con cada línea hablada ajustada a los límites de clips de 4/8/12 segundos de Sora.
- **Anchor, Reporter & Interviewee**: Más allá del presentador de estudio, el segmento incluye un **corresponsal** en ubicación y un **entrevistado** (un funcionario, experto o ciudadano), cada uno renderizado como su propio hablante frente a la cámara.
- **Native Speech & B-Roll**: Sora 2 genera audio y diálogo sincronizados directamente, por lo que cada hablante realmente habla. El productor intercala planos de b-roll sin rostros (paisajes urbanos, fábricas, granjas solares, mapas) para ilustrar el informe.
- **Scene Consistency (via remix)**: Sora 2 no tiene semilla y rechaza rostros humanos en `input_reference`, por lo que el *primer* clip de cada hablante es una generación nueva y cada clip posterior de ese mismo hablante es un **remix** de él — reutilizando la composición, vestuario e iluminación de la fuente para que el presentador/reportero/entrevistado permanezcan reconocibles. El b-roll sin rostros se encadena adicionalmente mediante último-encuadre→primer-encuadre (`input_reference`).
- **Seamless Stitching**: Todos los clips se fusionan con un `ffmpeg` estático empaquetado (vía `imageio-ffmpeg`, sin necesidad de paquete del sistema) en un único MP4 almacenado en Azure Blob Storage y transmitido de vuelta a la TV.
- **Non-Blocking Generation**: Tanto el texto de la decisión (tema + investigación web + redacción) como el video/narración se ejecutan en hilos de fondo protegidos por bloqueos de tabla de single-flight, por lo que la página nunca se queda colgada. Mientras se prepara el boletín de hoy, la TV muestra **ruido sin señal** ("Tuning in…") y vuelve a consultar `/get-string` hasta que el texto esté listo; luego cambia a video automáticamente una vez que el segmento esté listo. El estado se consulta vía `/genbox-video-status` y se almacena en caché por día.
- **CRT Controls, Now for Video**: El **botón inferior** alterna entre el texto de desplazamiento clásico y el video de noticias. Los botones anterior/siguiente se mueven a través de la línea de tiempo (cargando el video de cada día cuando esté disponible), y el botón de pausa pausa/reproduce el clip. Las decisiones más antiguas, solo en texto, continúan funcionando exactamente como antes.

#### Configuration
Video is generated for new dates only (configurable cutoff). Point GenBox at a dedicated Sora 2 deployment and storage container via `.env`:

```
AZURE_OPENAI_ENDPOINT_SORA       # Sora 2 resource endpoint(s)
AZURE_OPENAI_API_KEY_SORA        # matching API key(s)
AZURE_OPENAI_MODEL_SORA=sora-2   # deployment name(s)
AZURE_OPENAI_API_VERSION_SORA=preview
AZURE_OPENAI_MODEL_TTS=tts       # text-to-speech deployment on the SAME resources
genbox_video_blob_name=genbox-video   # blob container for merged MP4s + narration
GENBOX_VIDEO_CUTOFF_DATE=2026-06-05   # only dates >= this get video + narration
GENBOX_VIDEO_ENABLED=true
GENBOX_VIDEO_MAX_CLIPS=6               # cost cap on clips per segment
GENBOX_TTS_VOICE=onyx                 # narration voice (optional)
```

Cuando `AZURE_OPENAI_MODEL_TTS` está configurado, GenBox también narra la decisión diaria con un tono de portavoz gubernamental (vía el despliegue TTS de los mismos recursos). La narración se renderiza rápidamente, mucho antes que el video, y se reproduce sobre el texto de desplazamiento, con el desplazamiento ralentizado para coincidir con la duración del discurso. Está controlada, en caché, almacenada (en el contenedor blob de video) y referenciada en la tabla igual que el video, y se sirve del mismo origen vía `/genbox-audio`.

**Multiple Sora resources / distributed credits.** La API de Sora tiene alcance de trabajo (job-scoped): una llamada `create` devuelve un id de video que solo existe en el recurso que lo sirvió, por lo que la consulta posterior `poll {id}` y `download {id}` deben impactar ese *mismo* recurso; una puerta de enlace round-robin frente a varios recursos rompe esta afinidad. Para distribuir la carga entre recursos en su lugar, enuméralos **directamente** (no detrás de un balanceador) como valores separados por comas alineados por índice; GenBox aplica round-robin a nivel de *trabajo* y ancla todo el ciclo de vida create→poll→download de cada clip al recurso que seleccionó (y pasa un clip fallido al siguiente recurso en caso de error):

```
AZURE_OPENAI_ENDPOINT_SORA=https://res1.openai.azure.com,https://res2.openai.azure.com,https://res3.openai.azure.com
AZURE_OPENAI_API_KEY_SORA=key1,key2,key3
AZURE_OPENAI_MODEL_SORA=sora-2   # single value applies to all, or give one per resource
```

#### Tech Stack
- **OpenAI Agents SDK** (`openai-agents`) — el agente Producer que escribe el guion del segmento
- **Sora 2 on Azure OpenAI** — texto/imagen-a-video con audio nativo (`/openai/v1/videos`)
- **imageio-ffmpeg** — `ffmpeg` estático empaquetado para extracción del último encuadre y concatenación de clips
- **Azure Table & Blob Storage** — estado/metadata de video por fecha y alojamiento de MP4 fusionados

---
## Command to run the project
`gunicorn --bind=0.0.0.0 --timeout 3600 --workers 4 --threads 2 main:app`

(El tiempo de espera de 1 hora del trabajador coincide con el presupuesto de generación de ComicBook para que una ejecución de cómic larga no se corte; consulta `startup.sh`.)
