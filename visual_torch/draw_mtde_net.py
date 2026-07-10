import os
import sys
import torch.nn as nn
import visualtorch

# Asegurar que el directorio raíz del proyecto esté en sys.path para poder importar src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.mtde_net import MTDE_Net
from src.models.dstfs import SoftThresholdPReLU

def generate_diagrams():
    print("Inicializando modelo MTDE-Net...")
    model = MTDE_Net(tabular_dim=4)
    model.eval()

    input_shapes = ((1, 1, 112, 112), (1, 4))

    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Diagramas Originales (Detallados)
    print("Generando diagrama estilo 'flow' (detallado)...")
    try:
        img_flow = visualtorch.render(model=model, input_shape=input_shapes, style="flow")
        img_flow.save(os.path.join(output_dir, "mtde_net_flow.png"))
    except Exception as e:
        print(f"Error en flow detallado: {e}")

    print("Generando diagrama estilo 'graph' (detallado)...")
    try:
        img_graph = visualtorch.render(model=model, input_shape=input_shapes, style="graph")
        img_graph.save(os.path.join(output_dir, "mtde_net_graph.png"))
    except Exception as e:
        print(f"Error en graph detallado: {e}")

    # 2. Diagramas Simplificados (Excluyendo capas auxiliares)
    print("Generando diagrama estilo 'flow' simplificado (excluyendo BatchNorm, Activaciones, Dropout)...")
    try:
        # Excluimos capas de activación, normalización y auxiliares para dejar solo bloques estructurales
        layers_to_ignore = [
            nn.BatchNorm2d,
            nn.BatchNorm1d,
            nn.PReLU,
            nn.ReLU,
            nn.Dropout,
            nn.Flatten,
            nn.Identity,
            SoftThresholdPReLU
        ]
        
        img_flow_simple = visualtorch.render(
            model=model,
            input_shape=input_shapes,
            style="flow",
            type_ignore=layers_to_ignore,
            show_dimension=True  # Muestra las dimensiones en los bloques para mejor entendimiento
        )
        flow_simple_path = os.path.join(output_dir, "mtde_net_flow_simplified.png")
        img_flow_simple.save(flow_simple_path)
        print(f"Diagrama flow simplificado guardado en: {flow_simple_path}")
    except Exception as e:
        print(f"Error en flow simplificado: {e}")

if __name__ == "__main__":
    generate_diagrams()
