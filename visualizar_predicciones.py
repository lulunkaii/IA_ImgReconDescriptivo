import cv2
import torch
import numpy as np
from PIL import Image
import random
from torchvision import transforms, datasets
import torch.nn.functional as F
import time
import threading
import math

def slideshow_with_stats(model, dataset, device, num_samples=100, display_time=2):
    """
    Slideshow con estadísticas en tiempo real y mapas de atención
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
    
    print(f"Iniciando slideshow con estadísticas y mapas de atención...")
    print(f"Controles: 'q' = salir, 'p' = pausar, 'f' = más rápido, 's' = más lento")
    print(f"Secuencia: Imagen original ({display_time}s) -> Mapa de atención ({display_time}s) -> Siguiente imagen")
    
    paused = False
    current_time = display_time
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Obtener imagen y predicción
            image, true_label = dataset[idx]
            true_class = dataset.classes[true_label]
            
            image_tensor = image.unsqueeze(0).to(device)
            
            # Hacer predicción CON mapas de atención
            output, attn_maps = model(image_tensor, return_attn=True)
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
            
            # Preparar imagen base
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
            
            # Preparar mapa de atención
            last_attn = attn_maps[-1][0]  # [heads, tokens, tokens]
            cls_attn = last_attn[0, 0]    # atención del token CLS a todos los tokens
            patch_attn = cls_attn[1:]     # (196,) - solo los parches
            
            # Calcular grid_size basado en el número de parches
            num_patches = patch_attn.shape[0]
            grid_size = int(math.sqrt(num_patches))
            
            # Redimensionar el mapa de atención a la cuadrícula correcta
            patch_attn_grid = patch_attn.reshape(grid_size, grid_size).cpu().detach().numpy()
            
            # Redimensionar el mapa de atención a 400x400 para que coincida con la imagen
            patch_attn_resized = cv2.resize(patch_attn_grid, (400, 400), interpolation=cv2.INTER_CUBIC)
            
            # Normalizar el mapa de atención
            patch_attn_normalized = (patch_attn_resized - patch_attn_resized.min()) / (patch_attn_resized.max() - patch_attn_resized.min())
            
            # Crear colormap para el mapa de atención
            attention_colored = cv2.applyColorMap((patch_attn_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Crear imagen con mapa de atención superpuesto
            alpha = 0.6  # Transparencia del mapa de atención
            img_with_attention = cv2.addWeighted(img_cv, 1-alpha, attention_colored, alpha, 0)
            
            # Crear panel con estadísticas - IMAGEN ORIGINAL
            stats_width = 350
            display_img_original = np.ones((400, 400 + stats_width, 3), dtype=np.uint8) * 255
            display_img_original[:400, :400] = img_cv
            
            # Panel de estadísticas
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Título
            y_pos = 30
            cv2.putText(display_img_original, "ESTADISTICAS", (420, y_pos), 
                       font, font_scale + 0.1, (0, 0, 0), thickness + 1)
            
            # Estadísticas generales
            y_pos += 40
            accuracy = (stats['correct'] / stats['total']) * 100
            cv2.putText(display_img_original, f"Accuracy: {accuracy:.1f}%", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            y_pos += 25
            cv2.putText(display_img_original, f"Correctas: {stats['correct']}", (420, y_pos), 
                       font, font_scale, (0, 150, 0), thickness)
            
            y_pos += 20
            cv2.putText(display_img_original, f"Incorrectas: {stats['incorrect']}", (420, y_pos), 
                       font, font_scale, (0, 0, 150), thickness)
            
            y_pos += 20
            cv2.putText(display_img_original, f"Total: {stats['total']}", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Información actual
            y_pos += 40
            cv2.putText(display_img_original, "MUESTRA ACTUAL:", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            y_pos += 25
            cv2.putText(display_img_original, f"Real: {true_class}", (420, y_pos), 
                       font, font_scale - 0.1, (0, 0, 0), thickness)
            
            y_pos += 20
            cv2.putText(display_img_original, f"Pred: {predicted_class}", (420, y_pos), 
                       font, font_scale - 0.1, (0, 0, 0), thickness)
            
            y_pos += 20
            result_color = (0, 150, 0) if is_correct else (0, 0, 150)
            result_text = "CORRECTO" if is_correct else "INCORRECTO"
            cv2.putText(display_img_original, result_text, (420, y_pos), 
                       font, font_scale, result_color, thickness)
            
            # Información de confianza
            y_pos += 25
            cv2.putText(display_img_original, f"Confianza: {confidence.item():.3f}", (420, y_pos), 
                       font, font_scale - 0.1, (0, 0, 0), thickness)
            
            # Progreso
            y_pos += 30
            cv2.putText(display_img_original, f"Progreso: {i+1}/{len(indices)}", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Estado actual
            y_pos += 25
            cv2.putText(display_img_original, "MOSTRANDO: Imagen Original", (420, y_pos), 
                       font, font_scale - 0.1, (0, 100, 0), thickness)
            
            # Controles
            y_pos += 40
            cv2.putText(display_img_original, "CONTROLES:", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            controls = [
                "'q' - Salir",
                "'p' - Pausar",
                "'f' - Mas rapido",
                "'s' - Mas lento",
                "'n' - Siguiente"
            ]
            
            for control in controls:
                y_pos += 18
                cv2.putText(display_img_original, control, (420, y_pos), 
                           font, font_scale - 0.1, (100, 100, 100), thickness)
            
            # Mostrar velocidad actual
            y_pos += 25
            cv2.putText(display_img_original, f"Velocidad: {current_time:.1f}s", (420, y_pos), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Fase 1: Mostrar imagen original
            print(f"\nMuestra {i+1}/{len(indices)} (ID: {idx})")
            print(f"Verdadero: {true_class} | Predicción: {predicted_class}")
            print(f"Resultado: {'✓ CORRECTO' if is_correct else '✗ INCORRECTO'}")
            print(f"Confianza: {confidence.item():.3f}")
            print("Fase 1: Mostrando imagen original...")
            
            cv2.imshow("ViT Slideshow con Atención", display_img_original)
            
            # Esperar con controles - Fase 1
            start_time = time.time()
            phase1_interrupted = False
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
                    print(f"{'Pausado' if paused else 'Reanudado'}")
                elif key == ord('f'):
                    current_time = max(0.5, current_time - 0.5)
                    print(f"Velocidad: {current_time:.1f}s")
                elif key == ord('s'):
                    current_time = min(10.0, current_time + 0.5)
                    print(f"Velocidad: {current_time:.1f}s")
                elif key == ord('n'):  # Saltar a siguiente imagen
                    phase1_interrupted = True
                    break
                
                if paused:
                    continue
            
            # Fase 2: Mostrar mapa de atención (solo si no fue interrumpido)
            if not phase1_interrupted:
                print("Fase 2: Mostrando mapa de atención...")
                
                # Crear panel con estadísticas - IMAGEN CON ATENCIÓN
                display_img_attention = np.ones((400, 400 + stats_width, 3), dtype=np.uint8) * 255
                display_img_attention[:400, :400] = img_with_attention
                
                # Panel de estadísticas (igual que antes)
                y_pos = 30
                cv2.putText(display_img_attention, "ESTADISTICAS + ATENCION", (420, y_pos), 
                           font, font_scale + 0.1, (0, 0, 0), thickness + 1)
                
                # Estadísticas generales
                y_pos += 40
                accuracy = (stats['correct'] / stats['total']) * 100
                cv2.putText(display_img_attention, f"Accuracy: {accuracy:.1f}%", (420, y_pos), 
                           font, font_scale, (0, 0, 0), thickness)
                
                y_pos += 25
                cv2.putText(display_img_attention, f"Correctas: {stats['correct']}", (420, y_pos), 
                           font, font_scale, (0, 150, 0), thickness)
                
                y_pos += 20
                cv2.putText(display_img_attention, f"Incorrectas: {stats['incorrect']}", (420, y_pos), 
                           font, font_scale, (0, 0, 150), thickness)
                
                y_pos += 20
                cv2.putText(display_img_attention, f"Total: {stats['total']}", (420, y_pos), 
                           font, font_scale, (0, 0, 0), thickness)
                
                # Información actual
                y_pos += 40
                cv2.putText(display_img_attention, "MUESTRA ACTUAL:", (420, y_pos), 
                           font, font_scale, (0, 0, 0), thickness)
                
                y_pos += 25
                cv2.putText(display_img_attention, f"Real: {true_class}", (420, y_pos), 
                           font, font_scale - 0.1, (0, 0, 0), thickness)
                
                y_pos += 20
                cv2.putText(display_img_attention, f"Pred: {predicted_class}", (420, y_pos), 
                           font, font_scale - 0.1, (0, 0, 0), thickness)
                
                y_pos += 20
                result_color = (0, 150, 0) if is_correct else (0, 0, 150)
                result_text = "CORRECTO" if is_correct else "INCORRECTO"
                cv2.putText(display_img_attention, result_text, (420, y_pos), 
                           font, font_scale, result_color, thickness)
                
                # Información de confianza
                y_pos += 25
                cv2.putText(display_img_attention, f"Confianza: {confidence.item():.3f}", (420, y_pos), 
                           font, font_scale - 0.1, (0, 0, 0), thickness)
                
                # Progreso
                y_pos += 30
                cv2.putText(display_img_attention, f"Progreso: {i+1}/{len(indices)}", (420, y_pos), 
                           font, font_scale, (0, 0, 0), thickness)
                
                # Estado actual
                y_pos += 25
                cv2.putText(display_img_attention, "MOSTRANDO: Mapa de Atencion", (420, y_pos), 
                           font, font_scale - 0.1, (0, 0, 200), thickness)
                
                # Controles
                y_pos += 40
                cv2.putText(display_img_attention, "CONTROLES:", (420, y_pos), 
                           font, font_scale, (0, 0, 0), thickness)
                
                for control in controls:
                    y_pos += 18
                    cv2.putText(display_img_attention, control, (420, y_pos), 
                               font, font_scale - 0.1, (100, 100, 100), thickness)
                
                # Mostrar velocidad actual
                y_pos += 25
                cv2.putText(display_img_attention, f"Velocidad: {current_time:.1f}s", (420, y_pos), 
                           font, font_scale, (0, 0, 0), thickness)
                
                cv2.imshow("ViT Slideshow con Atención", display_img_attention)
                
                # Esperar con controles - Fase 2
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
                        print(f"{'Pausado' if paused else 'Reanudado'}")
                    elif key == ord('f'):
                        current_time = max(0.5, current_time - 0.5)
                        print(f"Velocidad: {current_time:.1f}s")
                    elif key == ord('s'):
                        current_time = min(10.0, current_time + 0.5)
                        print(f"Velocidad: {current_time:.1f}s")
                    elif key == ord('n'):  # Saltar a siguiente imagen
                        break
                    
                    if paused:
                        continue
    
    cv2.destroyAllWindows()
    return stats