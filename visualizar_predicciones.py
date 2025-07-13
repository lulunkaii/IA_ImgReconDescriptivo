import cv2
import torch
import numpy as np
from PIL import Image
import random
from torchvision import transforms, datasets
import torch.nn.functional as F
import time
import threading

# Función para slideshow con estadísticas
def slideshow_with_stats(model, dataset, device, num_samples=100, display_time=2):
    """
    Slideshow con estadísticas en tiempo real
    """
    model.eval()
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Estadísticas
    stats = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'class_correct': {},
        'class_total': {}
    }
    
    # Inicializar contadores por clase
    for class_name in dataset.classes:
        stats['class_correct'][class_name] = 0
        stats['class_total'][class_name] = 0
    
    print(f"Iniciando slideshow con estadísticas...")
    print(f"Controles: 'q' = salir, 'p' = pausar, 'f' = más rápido, 's' = más lento")
    
    paused = False
    current_time = display_time
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Obtener imagen y predicción
            image, true_label = dataset[idx]
            true_class = dataset.classes[true_label]
            
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_label = torch.max(probabilities, 1)
            predicted_class = dataset.classes[predicted_label.item()]
            
            # Actualizar estadísticas
            stats['total'] += 1
            stats['class_total'][true_class] += 1
            
            is_correct = predicted_class == true_class
            if is_correct:
                stats['correct'] += 1
                stats['class_correct'][true_class] += 1
            else:
                stats['incorrect'] += 1
            
            # Crear visualización
            img_np = image.permute(1, 2, 0).numpy()
            
            if hasattr(dataset, 'transform') and any('Normalize' in str(t) for t in dataset.transform.transforms):
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2023, 0.1994, 0.2010])
                img_np = img_np * std + mean
            
            img_np = np.clip(img_np, 0, 1)
            img_cv = (img_np * 255).astype(np.uint8)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # Redimensionar
            img_cv = cv2.resize(img_cv, (400, 400), interpolation=cv2.INTER_CUBIC)
            
            # Crear panel con estadísticas
            stats_width = 350
            display_img = np.ones((400, 400 + stats_width, 3), dtype=np.uint8) * 255
            display_img[:400, :400] = img_cv
            
            # Panel de estadísticas
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Título
            y_pos = 30
            cv2.putText(display_img, "ESTADISTICAS", (420, y_pos), 
                       font, font_scale + 0.1, (0, 0, 0), thickness + 1)
            
            # Estadísticas generales
            y_pos += 40
            accuracy = (stats['correct'] / stats['total']) * 100
            cv2.putText(display_img, f"Accuracy: {accuracy:.1f}%", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            y_pos += 25
            cv2.putText(display_img, f"Correctas: {stats['correct']}", (420, y_pos), 
                       font, font_scale, (0, 150, 0), thickness)
            
            y_pos += 20
            cv2.putText(display_img, f"Incorrectas: {stats['incorrect']}", (420, y_pos), 
                       font, font_scale, (0, 0, 150), thickness)
            
            y_pos += 20
            cv2.putText(display_img, f"Total: {stats['total']}", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Información actual
            y_pos += 40
            cv2.putText(display_img, "MUESTRA ACTUAL:", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            y_pos += 25
            cv2.putText(display_img, f"Real: {true_class}", (420, y_pos), 
                       font, font_scale - 0.1, (0, 0, 0), thickness)
            
            y_pos += 20
            cv2.putText(display_img, f"Pred: {predicted_class}", (420, y_pos), 
                       font, font_scale - 0.1, (0, 0, 0), thickness)
            
            y_pos += 20
            result_color = (0, 150, 0) if is_correct else (0, 0, 150)
            result_text = "CORRECTO" if is_correct else "INCORRECTO"
            cv2.putText(display_img, result_text, (420, y_pos), 
                       font, font_scale, result_color, thickness)
            
            # Progreso
            y_pos += 40
            cv2.putText(display_img, f"Progreso: {i+1}/{len(indices)}", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Controles
            y_pos += 60
            cv2.putText(display_img, "CONTROLES:", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            controls = [
                "'q' - Salir",
                "'p' - Pausar",
                "'f' - Más rápido",
                "'s' - Más lento"
            ]
            
            for control in controls:
                y_pos += 18
                cv2.putText(display_img, control, (420, y_pos), 
                           font, font_scale - 0.1, (100, 100, 100), thickness)
            
            # Mostrar velocidad actual
            y_pos += 25
            cv2.putText(display_img, f"Velocidad: {current_time:.1f}s", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            cv2.imshow("ViT Slideshow", display_img)
            
            # Esperar con controles
            start_time = time.time()
            while time.time() - start_time < current_time:
                if paused:
                    key = cv2.waitKey(100) & 0xFF
                else:
                    key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return stats
                elif key == ord('p'):
                    paused = not paused
                elif key == ord('f'):
                    current_time = max(0.5, current_time - 0.5)
                    print(f"Velocidad: {current_time:.1f}s")
                elif key == ord('s'):
                    current_time = min(10.0, current_time + 0.5)
                    print(f"Velocidad: {current_time:.1f}s")
                
                if paused:
                    continue
    
    cv2.destroyAllWindows()
    return stats
