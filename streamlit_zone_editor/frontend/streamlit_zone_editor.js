// Streamlit Zone Editor Component
(function() {
    'use strict';
    
    // Obtener datos de Streamlit Components v1
    // Los datos se pasan a través de window.argsFromParent o window.streamlitComponentValue
    function getStreamlitData() {
        // Método 1: argsFromParent (método estándar de Streamlit Components v1)
        if (window.argsFromParent && window.argsFromParent.length > 0) {
            const args = window.argsFromParent[0];
            if (args && typeof args === 'object') {
                return args;
            }
        }
        
        // Método 2: streamlitComponentValue (alternativo)
        if (typeof window.streamlitComponentValue !== 'undefined') {
            return window.streamlitComponentValue;
        }
        
        // Método 3: Intentar desde el contexto padre
        try {
            if (window.parent && window.parent.argsFromParent) {
                const parentArgs = window.parent.argsFromParent;
                if (parentArgs && parentArgs.length > 0) {
                    return parentArgs[0];
                }
            }
        } catch (e) {
            console.log('No se pudo acceder al contexto padre:', e);
        }
        
        return {};
    }
    
    // Obtener datos cuando el script se carga
    const streamlitData = getStreamlitData();
    const imgData = streamlitData.img_data || '';
    const existingZones = Array.isArray(streamlitData.existing_zones) ? streamlitData.existing_zones : [];
    const selectedTeam = streamlitData.selected_team || '';
    const selectedColor = streamlitData.selected_color || '#00A04A';
    const width = streamlitData.width || 700;
    
    // Debug: mostrar datos recibidos
    console.log('Datos recibidos de Streamlit:', {
        hasImgData: !!imgData,
        existingZonesCount: existingZones.length,
        selectedTeam: selectedTeam,
        selectedColor: selectedColor,
        width: width
    });
    
    // Variables globales
    let canvas, ctx, img;
    let isDrawing = false;
    let startX, startY, currentX, currentY;
    let rectangles = [];
    let currentRect = null;
    let canvasWidth = width;
    let canvasHeight = 0;
    
    // Inicializar cuando el DOM esté listo
    function init() {
        // Crear estructura HTML
        const root = document.getElementById('root');
        root.innerHTML = `
            <div class="editor-container">
                <h3 class="editor-header">🎨 Editor de Planos</h3>
                <div class="editor-controls">
                    <button class="control-btn" onclick="startDrawing()">✏️ Dibujar</button>
                    <button class="control-btn" onclick="clearLast()">🗑️ Borrar Último</button>
                    <button class="control-btn delete" onclick="clearAll()">🗑️ Borrar Todo</button>
                    <button class="control-btn save" onclick="saveZones()">💾 Guardar Zonas</button>
                </div>
                <div class="canvas-container">
                    <canvas id="drawingCanvas"></canvas>
                </div>
                <div class="status-panel">
                    <div class="coordinates">
                        <strong>Coordenadas:</strong><br>
                        <span id="coordsDisplay">X: 0, Y: 0</span>
                    </div>
                    <div class="zones-list" id="zonesList">
                        <strong>Zonas creadas:</strong>
                        <div id="zonesContainer"></div>
                    </div>
                </div>
            </div>
            <img id="sourceImage" src="data:image/png;base64,${imgData}" style="display:none">
        `;
        
        // Agregar estilos
        const style = document.createElement('style');
        style.textContent = `
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 10px;
                background: #f8f9fa;
            }
            .editor-container {
                max-width: ${canvasWidth}px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .editor-header {
                background: #00A04A;
                color: white;
                padding: 10px 15px;
                margin: 0;
                font-size: 16px;
            }
            .editor-controls {
                padding: 10px 15px;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }
            .control-btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 12px;
                flex: 1;
                min-width: 120px;
            }
            .control-btn:hover {
                background: #0056b3;
            }
            .control-btn.delete {
                background: #dc3545;
            }
            .control-btn.delete:hover {
                background: #c82333;
            }
            .control-btn.save {
                background: #28a745;
            }
            .control-btn.save:hover {
                background: #218838;
            }
            .canvas-container {
                position: relative;
                background: white;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 5px;
            }
            #drawingCanvas {
                display: block;
                cursor: crosshair;
                border: 1px solid #ccc;
                max-width: 100%;
            }
            .status-panel {
                padding: 10px 15px;
                background: #e9ecef;
                border-top: 1px solid #dee2e6;
                font-size: 12px;
            }
            .coordinates {
                font-family: monospace;
                background: #2b303b;
                color: #00ff00;
                padding: 8px;
                border-radius: 5px;
                margin: 5px 0;
                font-size: 11px;
            }
            .zones-list {
                max-height: 150px;
                overflow-y: auto;
                margin: 10px 0;
            }
            .zone-item {
                padding: 5px;
                margin: 2px 0;
                background: white;
                border-radius: 3px;
                font-size: 11px;
            }
        `;
        document.head.appendChild(style);
        
        // Inicializar canvas
        canvas = document.getElementById('drawingCanvas');
        ctx = canvas.getContext('2d');
        img = document.getElementById('sourceImage');
        
        // Cargar zonas existentes
        rectangles = existingZones.map(z => ({
            x: z.x || 0,
            y: z.y || 0,
            w: z.w || 0,
            h: z.h || 0,
            color: z.color || selectedColor,
            team: z.team || selectedTeam || 'Nueva Zona'
        }));
        
        // Inicializar cuando la imagen cargue
        img.onload = function() {
            const aspectRatio = img.naturalHeight / img.naturalWidth;
            canvasHeight = Math.round(canvasWidth * aspectRatio);
            
            canvas.width = canvasWidth;
            canvas.height = canvasHeight;
            
            drawImageAndZones();
            updateZonesList();
        };
        
        // Event listeners
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);
    }
    
    function drawImageAndZones() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        if (currentRect) {
            drawRectangle(currentRect);
        }
        
        rectangles.forEach(rect => {
            const scaleX = canvas.width / img.naturalWidth;
            const scaleY = canvas.height / img.naturalHeight;
            
            const canvasRect = {
                x: rect.x * scaleX,
                y: rect.y * scaleY,
                w: rect.w * scaleX,
                h: rect.h * scaleY,
                color: rect.color,
                team: rect.team
            };
            
            drawRectangle(canvasRect);
        });
    }
    
    function drawRectangle(rect) {
        ctx.strokeStyle = rect.color || selectedColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
        
        ctx.fillStyle = (rect.color || selectedColor) + '40';
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    }
    
    function startDrawing() {
        isDrawing = true;
        canvas.style.cursor = 'crosshair';
    }
    
    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.pageX - rect.left - window.pageXOffset);
        const y = (e.pageY - rect.top - window.pageYOffset);
        
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        return {
            x: x * scaleX,
            y: y * scaleY
        };
    }
    
    function handleMouseDown(e) {
        if (!isDrawing) return;
        
        const coords = getCanvasCoordinates(e);
        startX = coords.x;
        startY = coords.y;
        
        currentRect = {
            x: startX, y: startY, w: 0, h: 0,
            color: selectedColor,
            team: selectedTeam || 'Nueva Zona'
        };
    }
    
    function handleMouseMove(e) {
        const coords = getCanvasCoordinates(e);
        
        if (!isDrawing) {
            document.getElementById('coordsDisplay').textContent = 
                `X: ${Math.round(coords.x)}, Y: ${Math.round(coords.y)}`;
            return;
        }
        
        if (!currentRect) return;
        
        currentX = coords.x;
        currentY = coords.y;
        
        currentRect.w = currentX - startX;
        currentRect.h = currentY - startY;
        
        document.getElementById('coordsDisplay').textContent = 
            `X: ${Math.round(startX)}, Y: ${Math.round(startY)}, ` +
            `Ancho: ${Math.round(currentRect.w)}, Alto: ${Math.round(currentRect.h)}`;
        
        drawImageAndZones();
    }
    
    function handleMouseUp(e) {
        if (!isDrawing || !currentRect) return;
        
        if (Math.abs(currentRect.w) > 10 && Math.abs(currentRect.h) > 10) {
            const scaleX = img.naturalWidth / canvas.width;
            const scaleY = img.naturalHeight / canvas.height;
            
            const newRect = {
                x: Math.round(currentRect.x * scaleX),
                y: Math.round(currentRect.y * scaleY),
                w: Math.round(currentRect.w * scaleX),
                h: Math.round(currentRect.h * scaleY),
                color: selectedColor,
                team: selectedTeam || 'Nueva Zona'
            };
            
            rectangles.push(newRect);
            updateZonesList();
        }
        
        currentRect = null;
        isDrawing = false;
        canvas.style.cursor = 'default';
        drawImageAndZones();
    }
    
    function clearLast() {
        if (rectangles.length > 0) {
            rectangles.pop();
            drawImageAndZones();
            updateZonesList();
        }
    }
    
    function clearAll() {
        if (rectangles.length > 0) {
            if (confirm('¿Estás seguro de que quieres eliminar TODAS las zonas?')) {
                rectangles = [];
                drawImageAndZones();
                updateZonesList();
            }
        }
    }
    
    function updateZonesList() {
        const container = document.getElementById('zonesContainer');
        container.innerHTML = '';
        
        rectangles.forEach((rect, index) => {
            const zoneDiv = document.createElement('div');
            zoneDiv.className = 'zone-item';
            zoneDiv.style.borderLeft = `3px solid ${rect.color}`;
            zoneDiv.innerHTML = `${index + 1}. ${rect.team} (${Math.round(rect.x)}, ${Math.round(rect.y)})`;
            container.appendChild(zoneDiv);
        });
    }
    
    function saveZones() {
        // Enviar datos a Streamlit usando la API de Streamlit Components v1
        const dataToSend = {
            zones: rectangles,
            action: 'save'
        };
        
        // Streamlit Components v1 usa window.parent.streamlit.setComponentValue
        try {
            // Método 1: API estándar de Streamlit Components
            if (window.parent && window.parent.streamlit && window.parent.streamlit.setComponentValue) {
                window.parent.streamlit.setComponentValue(dataToSend);
                alert('✅ Zonas guardadas automáticamente! (' + rectangles.length + ' zonas)');
                return;
            }
            
            // Método 2: Usar Streamlit global (si está disponible)
            if (window.Streamlit && typeof window.Streamlit.setComponentValue === 'function') {
                window.Streamlit.setComponentValue(dataToSend);
                alert('✅ Zonas guardadas automáticamente! (' + rectangles.length + ' zonas)');
                return;
            }
            
            // Método 3: Usar postMessage como fallback
            if (window.parent) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: JSON.stringify(dataToSend)
                }, '*');
                alert('✅ Zonas guardadas! (' + rectangles.length + ' zonas)');
                return;
            }
        } catch (e) {
            console.error('Error al guardar:', e);
        }
        
        alert('⚠️ No se pudo guardar automáticamente. Por favor, recarga la página.');
    }
    
    // Hacer funciones globales para los botones
    window.startDrawing = startDrawing;
    window.clearLast = clearLast;
    window.clearAll = clearAll;
    window.saveZones = saveZones;
    
    // Inicializar cuando el DOM esté listo
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

