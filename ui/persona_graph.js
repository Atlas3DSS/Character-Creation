/* Orbitable 3D projections of measured profiles and checkpoint affinities.
   No invented physical topology: coordinates and edge values come from the API. */
(()=>{
'use strict';
const cameras=new Map(),previous=new Map();
const node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
function render(graph,options={}){
 const compact=!!options.compact,layered=options.layout==='layers',key=(compact?'small:':'large:')+(graph?.directed?'weights:':'activity:')+(options.layout||'similarity');
 const root=node('div');root.className=compact?'pg-scene pg-compact':'pg-scene';Object.assign(root.style,{position:'relative',overflow:'hidden',isolation:'isolate',contain:'layout paint',width:'100%',boxSizing:'border-box'});
 const canvas=node('canvas');canvas.className='pg-canvas';Object.assign(canvas.style,{position:'relative',inset:'auto',display:'block',width:'100%',maxWidth:'100%',touchAction:'none'});canvas.tabIndex=compact?-1:0;canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Interactive 3D connection graph. Drag to orbit, scroll to zoom, arrow keys to rotate, plus or minus to zoom, R to reset.');
 root.append(canvas);
 if(!graph?.nodes?.length){root.append(node('p','No supported graph data'));return root;}
 const nodes=graph.nodes,ids=new Set(nodes.map(n=>n.id));
 const edges=(graph.edges||[]).filter(e=>ids.has(e.source)&&ids.has(e.target)&&Math.abs(e.r??e.score)>=Number(options.threshold??.6));
 const degrees=new Map(nodes.map(n=>[n.id,0]));edges.forEach(e=>{degrees.set(e.source,degrees.get(e.source)+1);degrees.set(e.target,degrees.get(e.target)+1);});
 const raw=nodes.map(n=>layered?[n.position[0],n.position[1],n.layer]:options.layout==='topology'?n.topology3d:n.position3d);
 if(raw.some(p=>!p||p.length!==3||!p.every(Number.isFinite))){root.append(node('p','Missing three-dimensional measurement coordinates.'));return root;}
 const bounds=[0,1,2].map(a=>[Math.min(...raw.map(p=>p[a])),Math.max(...raw.map(p=>p[a]))]);
 const common=Math.max(...bounds.map(([a,b])=>b-a),1e-9);
 const coords=new Map(nodes.map((n,i)=>[n.id,raw[i].map((v,a)=>{
   const span=layered?Math.max(bounds[a][1]-bounds[a][0],1e-9):common;
   return 2*(v-(bounds[a][0]+bounds[a][1])/2)/span;
 })]));
 const camera=cameras.get(key)||{yaw:-.55,pitch:.25,zoom:1,center:[0,0,0]};camera.q=camera.q||[1,0,0,0];cameras.set(key,camera);
 let drag=null,hover=null,points=[],width=700,height=420,animation=0,showAll=false,spread=1.7,gizmoMode='rotate',gizmoHits=[],gizmoDrag=null;
 const selectedId=nodes.find(n=>n.selected)?.id||nodes[0].id;
 const visibleEdges=()=>showAll?edges:edges.filter(e=>e.source===selectedId||e.target===selectedId);
 const context=canvas.getContext('2d'),start=performance.now(),duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:450;
 const from=new Map(nodes.map(n=>[n.id,previous.get(key+n.id)||coords.get(n.id)]));
 const edgeScale=options.normalizeEdges?Math.max(...edges.map(e=>Math.abs(e.r??e.score)),1e-9):1;
 const status=node('p');status.className='pg-status';
 const reset=()=>{Object.assign(camera,{yaw:-.55,pitch:.25,zoom:1,center:[0,0,0],q:[1,0,0,0]});draw();};
 if(!compact){
  const controls=node('div');controls.className='pg-controls';
  const make=(text,fn)=>{const b=node('button',text);b.type='button';b.onclick=fn;controls.append(b);};
  make('Reset 3D view',reset);
  const all=node('button','Show neighbor-to-neighbor links');all.type='button';all.setAttribute('aria-pressed','false');all.onclick=()=>{showAll=!showAll;all.textContent=showAll?'Show selected neuron links only':'Show neighbor-to-neighbor links';all.setAttribute('aria-pressed',String(showAll));draw();};controls.append(all);
  const space=node('label','Visual spacing '),slider=node('input');slider.type='range';slider.min='1';slider.max='3';slider.step='.1';slider.value='1.7';slider.setAttribute('aria-label','Visual spacing (display distortion)');slider.oninput=()=>{spread=Number(slider.value);draw();};space.append(slider);controls.append(space);
make('Focus selected',()=>{const selected=nodes.find(n=>n.selected);if(selected){camera.center=expanded(coords.get(selected.id));camera.zoom=1.3;draw();}});
  const gizmo=node('select');gizmo.setAttribute('aria-label','3D gizmo mode');for(const [value,label] of [['rotate','Gizmo: rotate'],['move','Gizmo: move'],['off','Gizmo: hidden']]){const option=node('option',label);option.value=value;gizmo.append(option);}gizmo.onchange=()=>{gizmoMode=gizmo.value;draw();};controls.append(gizmo);
  for(let axis=0;axis<3;axis++){make('↻ '+['X','Y','Z'][axis],()=>{spin(axis,.15);draw();});}
  const pick=node('select');pick.setAttribute('aria-label','Select a graph neuron');nodes.forEach(n=>{const o=node('option','L'+n.layer+' N'+n.neuron);o.value=n.id;pick.append(o);});pick.value=nodes.find(n=>n.selected)?.id||nodes[0].id;
  pick.onchange=()=>options.onSelect?.(nodes.find(n=>n.id===pick.value));controls.append(pick);root.append(controls,status);
  const detail=node('details'),summary=node('summary','Exact relationships to selected neuron'),table=node('table');detail.append(summary,table);
  for(const e of edges.filter(e=>e.source===selectedId||e.target===selectedId).sort((a,b)=>Math.abs(b.r??b.score)-Math.abs(a.r??a.score))){const row=node('tr'),id=e.source===selectedId?e.target:e.source,cell=node('td'),b=node('button',id);b.type='button';b.onclick=()=>options.onSelect?.(nodes.find(n=>n.id===id));cell.append(b);row.append(cell,node('td',(graph.directed?(e.source===selectedId?'outgoing':'incoming'):'Pearson r')),node('td',(e.r??e.score).toFixed(4)));table.append(row);}root.append(detail);
  status.textContent='Drag to orbit · wheel/pinch to zoom · click a node · cyan = positive · pink dashed = negative · spacing distorts display only';
 }
 function expanded(p){const length=Math.hypot(...p);return spread>1&&length>1e-9?p.map(v=>v*Math.pow(length/Math.sqrt(3),1/spread-1)):p;}
 function multiply(a,b){return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];}
 function spin(axis,angle){const r=[Math.cos(angle/2),0,0,0];r[axis+1]=Math.sin(angle/2);camera.q=multiply(camera.q,r);const norm=Math.hypot(...camera.q);camera.q=camera.q.map(v=>v/norm);}
 function rotated(p){
  const q=camera.q,r=multiply(multiply(q,[0,...p]),[q[0],-q[1],-q[2],-q[3]]).slice(1);
  const [x,y,z]=r.map((v,a)=>v-camera.center[a]),cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
  const xx=x*cy+z*sy,zz=-x*sy+z*cy;
  return [xx,y*cp-zz*sp,y*sp+zz*cp];
 }
 function project(p){
  const [x,y,z]=rotated(p),perspective=4.8/Math.max(1.8,4.8-z),scale=Math.min(width,height)*.29*camera.zoom*perspective;
  return {x:width/2+x*scale,y:height/2+y*scale,z,scale:perspective};
 }
 function line(a,b){context.beginPath();context.moveTo(a.x,a.y);context.lineTo(b.x,b.y);context.stroke();}
 function draw(now=performance.now()){
  const rect=canvas.getBoundingClientRect();width=rect.width||700;height=rect.height||(compact?120:420);
  const dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
  context.setTransform(dpr,0,0,dpr,0,0);context.clearRect(0,0,width,height);
  const t=duration?Math.min(1,(now-start)/duration):1,ease=1-Math.pow(1-t,3);
  const positions=new Map(nodes.map(n=>{const a=from.get(n.id),b=coords.get(n.id);let p=a.map((v,k)=>v+(b[k]-v)*ease);p=expanded(p);return [n.id,p];}));
  const shown=visibleEdges(),focus=hover?.id||selectedId;
  if(layered&&!compact){
   const layers=[...new Set(nodes.map(n=>n.layer))];
   for(const l of layers.sort((a,b)=>project([0,0,coords.get(nodes.find(n=>n.layer===a).id)[2]]).z-project([0,0,coords.get(nodes.find(n=>n.layer===b).id)[2]]).z)){
    const z=coords.get(nodes.find(n=>n.layer===l).id)[2],corners=[[-1.12,-1.12,z],[1.12,-1.12,z],[1.12,1.12,z],[-1.12,1.12,z]].map(project);
    context.beginPath();context.moveTo(corners[0].x,corners[0].y);corners.slice(1).forEach(p=>context.lineTo(p.x,p.y));context.closePath();context.fillStyle='rgba(134,194,50,.025)';context.fill();context.strokeStyle='rgba(142,162,173,.13)';context.lineWidth=1;context.stroke();context.fillStyle='#94a5ad';context.font='11px monospace';context.fillText('L'+l,corners[0].x,corners[0].y-5);
   }
  }
  const projected=new Map([...positions].map(([id,p])=>[id,project(p)])),bundle=Number(options.bundle??.2);
  for(const edge of [...shown].sort((a,b)=>(projected.get(a.source).z+projected.get(a.target).z)-(projected.get(b.source).z+projected.get(b.target).z))){
   const a=positions.get(edge.source),b=positions.get(edge.target),r=edge.r??edge.score,related=edge.source===focus||edge.target===focus;
   context.globalAlpha=related?.85:.08;context.strokeStyle=r<0?'#f196bc':graph.directed||r>.2?'#63d7df':'#aab9c1';
   context.lineWidth=(.55+2.7*Math.min(1,Math.abs(r)/edgeScale))*(related&&hover?1.3:1);context.setLineDash(r<0?[5,3]:[]);
   const control=a.map((v,k)=>(v+b[k])/2*(1-bundle)),curve=[];
   for(let j=0;j<=18;j++){const q=j/18;curve.push(project(a.map((v,k)=>(1-q)*(1-q)*v+2*q*(1-q)*control[k]+q*q*b[k])));}
   context.beginPath();curve.forEach((p,i)=>i?context.lineTo(p.x,p.y):context.moveTo(p.x,p.y));context.stroke();context.setLineDash([]);
   if(hover&&related&&!compact){const mid=curve[9];context.font='12px monospace';const label=(graph.directed?'cos ':'r ')+(r>=0?'+':'')+r.toFixed(3);context.fillStyle='#101719';context.fillRect(mid.x-3,mid.y-13,context.measureText(label).width+6,17);context.fillStyle='#f2f5f6';context.fillText(label,mid.x,mid.y);}
   if(graph.directed){const tip=curve[15],before=curve[13],angle=Math.atan2(tip.y-before.y,tip.x-before.x);context.fillStyle=context.strokeStyle;context.beginPath();context.moveTo(tip.x,tip.y);context.lineTo(tip.x-7*Math.cos(angle-.45),tip.y-7*Math.sin(angle-.45));context.lineTo(tip.x-7*Math.cos(angle+.45),tip.y-7*Math.sin(angle+.45));context.closePath();context.fill();}
  }
  context.globalAlpha=1;
  points=nodes.map(n=>({...projected.get(n.id),n,radius:(compact?2.4:4)+Math.min(6,Math.sqrt(degrees.get(n.id))*.65)})).sort((a,b)=>a.z-b.z);
  for(const p of points){
   const n=p.n,r=p.radius*p.scale,isHover=hover?.id===n.id;
   context.globalAlpha=hover&&!isHover&&!edges.some(e=>(e.source===hover.id&&e.target===n.id)||(e.target===hover.id&&e.source===n.id))?.22:Math.max(.45,Math.min(1,.7+p.z*.12));
   const fill=n.selected?'#efc77c':n.r<0?'#de90b4':'#86c232';
   context.beginPath();context.arc(p.x,p.y,r,0,Math.PI*2);context.fillStyle=fill;context.shadowColor=fill;context.shadowBlur=isHover?12:0;context.fill();context.shadowBlur=0;
   context.strokeStyle='#101719';context.lineWidth=1;context.stroke();
   if(n.selected||isHover){context.globalAlpha=1;context.beginPath();context.arc(p.x,p.y,r+4,0,Math.PI*2);context.strokeStyle=fill;context.stroke();if(!compact){context.font='11px monospace';context.fillStyle='#e6edef';context.fillText(n.id,p.x+r+7,p.y+4);}}
  }
  context.globalAlpha=1;gizmoHits=[];
  if(!compact&&gizmoMode!=='off'){
   const knob=p=>{const old=camera.center;camera.center=[0,0,0];const r=rotated(p);camera.center=old;return {x:width-74+r[0]*43,y:height-74+r[1]*43,z:r[2]};};
   context.fillStyle='rgba(16,23,25,.88)';context.beginPath();context.arc(width-74,height-74,68,0,Math.PI*2);context.fill();
   const center=knob([0,0,0]),colors=['#ff7777','#87dc74','#74b6ff'];
   for(let axis=0;axis<3;axis++){
    context.strokeStyle=colors[axis];context.fillStyle=colors[axis];context.lineWidth=2;context.globalAlpha=gizmoDrag?.axis===axis?1:.65;
    if(gizmoMode==='rotate'){
     const ring=[];for(let j=0;j<=80;j++){const v=[0,0,0],a=j/80*Math.PI*2;v[(axis+1)%3]=1.22*Math.cos(a);v[(axis+2)%3]=1.22*Math.sin(a);ring.push(knob(v));}
     context.beginPath();ring.forEach((p,i)=>{if(i){context.lineTo(p.x,p.y);gizmoHits.push({a:ring[i-1],b:p,axis});}else context.moveTo(p.x,p.y);});context.stroke();
     const label=ring[10+axis*10];context.font='bold 13px monospace';context.fillText(['X','Y','Z'][axis],label.x+5,label.y-5);
    }else{
     const v=[0,0,0];v[axis]=1.4;const tip=knob(v);line(center,tip);gizmoHits.push({a:center,b:tip,axis});const angle=Math.atan2(tip.y-center.y,tip.x-center.x);context.beginPath();context.moveTo(tip.x,tip.y);context.lineTo(tip.x-11*Math.cos(angle-.4),tip.y-11*Math.sin(angle-.4));context.lineTo(tip.x-11*Math.cos(angle+.4),tip.y-11*Math.sin(angle+.4));context.closePath();context.fill();context.fillText(['X','Y','Z'][axis],tip.x+8,tip.y);
    }
   }
  }
  context.globalAlpha=1;
  if(!compact){
   const origin={x:34,y:height-35};const axes=[[[.22,0,0],'#de90b4',layered?'write x':options.layout==='topology'?'graph1':'PC1'],[[0,.22,0],'#86c232',layered?'write y':options.layout==='topology'?'graph2':'PC2'],[[0,0,.22],'#8fcbe3',layered?'layer':options.layout==='topology'?'graph3':'PC3']];
   axes.forEach(([v,color,label])=>{const old=camera.center;camera.center=[0,0,0];const p=rotated(v);camera.center=old;const to={x:origin.x+p[0]*100,y:origin.y+p[1]*100};context.strokeStyle=color;context.lineWidth=2;line(origin,to);context.fillStyle=color;context.font='9px monospace';context.fillText(label,to.x+2,to.y);});
  }
  if(t<1&&root.isConnected)animation=requestAnimationFrame(draw);
  else{nodes.forEach(n=>previous.set(key+n.id,coords.get(n.id)));while(previous.size>4096)previous.delete(previous.keys().next().value);}
 }
 const shownCount=id=>visibleEdges().filter(e=>e.source===id||e.target===id).length;
 const hit=(x,y)=>[...points].reverse().find(p=>Math.hypot(p.x-x,p.y-y)<Math.max(10,p.radius*p.scale+4));
 function gizmoHit(x,y){return gizmoHits.map(e=>{const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,t=Math.max(0,Math.min(1,((x-e.a.x)*dx+(y-e.a.y)*dy)/(dx*dx+dy*dy||1)));return {...e,d:Math.hypot(x-e.a.x-t*dx,y-e.a.y-t*dy)};}).filter(e=>e.d<7).sort((a,b)=>a.d-b.d)[0];}
 const pointers=new Map();let pinch=null;
 canvas.onpointerdown=e=>{if(!compact)e.preventDefault();const rect=canvas.getBoundingClientRect(),g=gizmoHit(e.clientX-rect.left,e.clientY-rect.top);gizmoDrag=g?{axis:g.axis,x:e.clientX,y:e.clientY}:null;canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);drag={id:e.pointerId,x:e.clientX,y:e.clientY,moved:0};if(pointers.size===2){const p=[...pointers.values()];pinch=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);}};
 canvas.onpointermove=e=>{
  if(gizmoDrag&&drag?.id===e.pointerId){const dx=e.clientX-gizmoDrag.x,dy=e.clientY-gizmoDrag.y;drag.moved+=Math.abs(dx)+Math.abs(dy);if(gizmoMode==='rotate')spin(gizmoDrag.axis,(dx-dy)*.01);else{const a=project([0,0,0]),v=[0,0,0];v[gizmoDrag.axis]=1;const b=project(v),sx=b.x-a.x,sy=b.y-a.y,delta=(dx*sx+dy*sy)/(sx*sx+sy*sy||1),q=camera.q,world=multiply(multiply(q,[0,...v]),[q[0],-q[1],-q[2],-q[3]]).slice(1);camera.center=camera.center.map((value,i)=>value-world[i]*delta);}gizmoDrag.x=e.clientX;gizmoDrag.y=e.clientY;draw();return;}
  if(pointers.has(e.pointerId)){pointers.set(e.pointerId,[e.clientX,e.clientY]);if(pointers.size===2){const p=[...pointers.values()],distance=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);if(pinch)camera.zoom=Math.max(.4,Math.min(2.8,camera.zoom*distance/pinch));pinch=distance;if(drag)drag.moved=100;draw();return;}}
  if(drag&&drag.id===e.pointerId){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved+=Math.abs(dx)+Math.abs(dy);camera.yaw+=dx*.007;camera.pitch=Math.max(-1.5,Math.min(1.5,camera.pitch+dy*.007));drag.x=e.clientX;drag.y=e.clientY;hover=null;draw();return;}
  const rect=canvas.getBoundingClientRect(),p=hit(e.clientX-rect.left,e.clientY-rect.top);hover=p?.n||null;canvas.style.cursor=hover?'pointer':'grab';if(hover){status.textContent=hover.id+' · '+shownCount(hover.id)+' visible connections · '+edges.filter(e=>e.source===hover.id||e.target===hover.id).sort((a,b)=>Math.abs(b.r??b.score)-Math.abs(a.r??a.score)).slice(0,4).map(e=>(e.source===hover.id?e.target:e.source)+': '+(e.r??e.score).toFixed(3)).join(' · ');canvas.title=status.textContent;}else{status.textContent='Drag to orbit · wheel/pinch to zoom · click a node · cyan = positive · pink dashed = negative · spacing distorts display only';canvas.title='';}draw();
 };
 const release=e=>{const clicked=!gizmoDrag&&drag?.id===e.pointerId&&drag.moved<5&&e.type==='pointerup';pointers.delete(e.pointerId);pinch=null;drag=null;gizmoDrag=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);if(clicked){const rect=canvas.getBoundingClientRect(),p=hit(e.clientX-rect.left,e.clientY-rect.top);if(p)options.onSelect?.(p.n);}};
 canvas.onpointerup=canvas.onpointercancel=canvas.onlostpointercapture=release;
 canvas.onpointerleave=()=>{if(!drag){hover=null;draw();}};
 canvas.onwheel=e=>{e.preventDefault();camera.zoom=Math.max(.4,Math.min(2.8,camera.zoom*Math.exp(-e.deltaY*.001)));draw();};
 canvas.onkeydown=e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','r','R'].includes(e.key)){e.preventDefault();if(e.key==='ArrowLeft')camera.yaw-=.12;if(e.key==='ArrowRight')camera.yaw+=.12;if(e.key==='ArrowUp')camera.pitch-=.12;if(e.key==='ArrowDown')camera.pitch+=.12;if(e.key==='+')camera.zoom=Math.min(2.8,camera.zoom*1.1);if(e.key==='-')camera.zoom=Math.max(.4,camera.zoom/1.1);if(e.key.toLowerCase()==='r')reset();draw();}};
 const observer=new ResizeObserver(()=>{if(root.isConnected)draw();else observer.disconnect();});observer.observe(canvas);
 requestAnimationFrame(draw);return root;
}
window.PersonaGraph={render};
})();
