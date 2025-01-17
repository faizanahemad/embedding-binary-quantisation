We want to build two diagrams in methodology.tex.

1. The first diagram should show the overall architecture of QAMA.
2. The second diagram should show the hybrid quantization architecture of QAMA.

Tenets:
- Diagrams should be horizontal.
- Diagrams should be simple and easy to understand.
- Diagrams should convey our methodology and solution.
- Use different colors for different components. Use one color for the same component in different diagrams.
- Use one color for trainable vs another color for non-trainable components.
- Use one color for components that are used in training vs another color for components that are used in inference.
- Make use of arrows and simple shapes like rounded rectangles, circles, ellipses, etc. Write the text on the shapes inside them.

Details on Diagram 1:

It will be a horizontal diagram with the following components:

- Transformer-based sentence/embedding encoder
- FFN Layer
- Matryoshka Representation Learning (MRL), russian doll like diagram.
- Associated Loss Functions on MRL
- Quantization Losses
- Other Losses
- Quantization threshold finding with momentum
- Bit Expansion for Hamming Distance
- Bit-packing

Details on Diagram 2:

It will be a horizontal diagram with the following components:

- Transformer-based sentence/embedding encoder
- FFN Layer
- Matryoshka Representation Learning (MRL), russian doll like diagram.
- Associated Loss Functions on MRL
- Quantization Losses
- Other Losses
- Fork into 4 branches for hybrid quantization
- Quantization threshold finding with momentum
- Bit Expansion for Hamming Distance
- Bit-packing

These diagrams will be drawn using draw.io.

**Diagramming and Plotting Instructions**
- First Decide if you need to make a diagram or not. If you need to make a diagram, then decide if you need to make a mermaid diagram or a draw.io diagram or a matplotlib or seaborn plot.
- Mermaid diagrams can be made using mermaid js library syntax. Write the mermaid diagram code inside <pre class="mermaid"> and </pre> tags.
- You can make Flowcharts, Sequence Diagrams, Gantt diagram, Class diagram, User Journey Diagram, Quadrant Chart, XY Chart. Write the diagram code inside <pre class="mermaid"> and </pre> tags so that our mermaid parser can pick it and draw it.
- You are allowed to make diagrams using draw.io or diagrams.net xml format. Always Write the draw.io xml code inside triple ticks like (```xml <Drawio xml code> ```).
- Use draw.io or diagrams.net to make diagrams like System design diagrams, complex scientific processes, flowcharts, network diagrams, architecture diagrams etc. Always Write the draw.io xml code inside triple ticks like (```xml <Drawio xml code> ```). so that our drawio parser can pick it and draw it.
- Diagrams, charts, flow diagrams, sequence diagrams, Gantt diagrams, class diagrams, and other graphic representations are very effective in helping the user understand the problem and solution, as well as in helping the user learn the solution.

