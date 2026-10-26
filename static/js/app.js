const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const conversationStorageKey = 'agentic-rag-conversation-id';
let conversationId = null;
let conversationReady = Promise.resolve();
let isSending = false;

function escapeHtml(s=''){return s.replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
async function readApiResponse(res){
  const contentType=(res.headers.get('content-type')||'').toLowerCase();
  if(contentType.includes('application/json')){
    try{return await res.json();}catch{}
  }
  return {detail:`Server returned a non-JSON response (HTTP ${res.status}). Check the API route and server logs.`};
}
function apiError(res,data,fallback='Request failed'){
  let message=data.detail||fallback;
  const requestId=res.headers.get('x-request-id');
  if(requestId&&!String(message).includes(requestId)) message+=` (Request ID: ${requestId})`;
  return new Error(message);
}
function addMessage(role, text, source='', citations=[]){
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>`${c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)}${c.excerpt?`<blockquote>${escapeHtml(c.excerpt)}</blockquote>`:''}`).join('<br>')}</div>`:'';
  wrap.innerHTML=`<div class="avatar">${role==='assistant'?'AI':'You'}</div><div class="bubble">${formatText(text)}${source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:''}${citeHtml}</div>`;
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}
function showWelcome(){chat.innerHTML='';addMessage('assistant','How can I help with IT support? Try an IT support question or ask about current vendor information.');}
async function createConversation(){
  const res=await fetch('/api/conversations',{method:'POST'}); const data=await readApiResponse(res);
  if(!res.ok) throw apiError(res,data,'Could not start a conversation');
  conversationId=data.conversation_id; localStorage.setItem(conversationStorageKey,conversationId);
  showWelcome(); renderTrace([]); sourceUsed.textContent='—';
}
async function resumeConversation(){
  const saved=localStorage.getItem(conversationStorageKey);
  if(!saved){await createConversation();return;}
  const res=await fetch(`/api/conversations/${encodeURIComponent(saved)}`);
  if(res.status===404){localStorage.removeItem(conversationStorageKey);await createConversation();return;}
  const data=await readApiResponse(res);
  if(!res.ok) throw apiError(res,data,'Could not resume the conversation');
  conversationId=data.conversation_id; chat.innerHTML='';
  if(!data.messages.length) showWelcome(); else data.messages.forEach(m=>addMessage(m.role,m.content));
}
async function askAgent(q){
  await conversationReady;
  if(isSending) return;
  isSending=true;
  const btn=form.querySelector('button'); const oldButtonText=btn.innerHTML; btn.disabled=true; btn.innerHTML='<span>Thinking</span><span aria-hidden="true">...</span>'; form.setAttribute('aria-busy','true');
  document.querySelectorAll('.example').forEach(button=>button.disabled=true);
  try{
    if(!conversationId) await createConversation();
    addMessage('user',q); question.value=''; question.style.height=''; renderTrace(['Looking through conversation context and trusted sources...']); sourceUsed.textContent='Working';
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,conversation_id:conversationId})});
    const data=await readApiResponse(res); if(!res.ok) throw apiError(res,data);
    conversationId=data.conversation_id; localStorage.setItem(conversationStorageKey,conversationId);
    addMessage('assistant',data.answer,data.source_used,data.citations||[]); renderTrace(data.trace||[]); sourceUsed.textContent=data.source_used;
  }catch(e){addMessage('assistant',`Error: ${e.message}`); renderTrace(['Request failed']); sourceUsed.textContent='Error';}
  finally{btn.disabled=false; btn.innerHTML=oldButtonText; form.removeAttribute('aria-busy'); document.querySelectorAll('.example').forEach(button=>button.disabled=false); isSending=false; question.focus();}
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
question.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();form.requestSubmit();}});
question.addEventListener('input',()=>{question.style.height='auto';question.style.height=`${Math.min(question.scrollHeight,180)}px`;});
document.querySelectorAll('.example').forEach(b=>b.addEventListener('click',()=>askAgent(b.textContent.trim())));
document.getElementById('newConversation').addEventListener('click',async()=>{
  await conversationReady;
  try{await createConversation();}catch(e){addMessage('assistant',`Error: ${e.message}`);}
});
conversationReady=resumeConversation().catch(e=>addMessage('assistant',`Error: ${e.message}`));

const modal=document.getElementById('uploadModal');
const openUploadButton=document.getElementById('openUpload');
function closeUploadDialog(){modal.classList.add('hidden');openUploadButton.focus();}
openUploadButton.onclick=()=>{modal.classList.remove('hidden');document.getElementById('adminKey').focus();};
document.getElementById('closeUpload').onclick=closeUploadDialog;
modal.addEventListener('click',event=>{if(event.target===modal)closeUploadDialog();});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!modal.classList.contains('hidden'))closeUploadDialog();});
document.getElementById('uploadBtn').onclick=async()=>{
  const file=document.getElementById('fileInput').files[0]; const key=document.getElementById('adminKey').value; const status=document.getElementById('uploadStatus');
  if(!file){status.textContent='Choose a file first.'; return;}
  const button=document.getElementById('uploadBtn');
  button.disabled=true; button.textContent='Indexing...'; status.textContent=`Preparing ${file.name} for indexing...`;
  const fd=new FormData(); fd.append('file',file);
  try{
    const r=await fetch('/api/ingest',{method:'POST',headers:{'X-Admin-Key':key},body:fd});
    const d=await r.json(); if(!r.ok) throw new Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||'Upload failed'));
    const warningText=d.warnings?.length?`\nWarnings: ${d.warnings.join(' ')}`:'';
    const visionStatus=d.vision_analysis_status||d.vision_status||'unknown';
    const visionLabel={completed:'complete',partial:'partial',failed:'failed',not_configured:'not configured',not_needed:'not needed'}[visionStatus]||visionStatus;
    const textLabel=d.text_ocr_index_status==='completed'?`${d.text_ocr_chunks} text/OCR chunk(s) indexed`:'no text/OCR chunks';
    status.textContent=`Indexed ${d.file}: ${d.chunks} chunks from ${d.pages} page(s). Searchable content indexing: ${d.document_index_status||'completed'}. ${textLabel}. Visual analysis: ${visionLabel}.${warningText}`;
  }catch(e){status.textContent=`Upload failed: ${e.message}`;}
  finally{button.disabled=false; button.textContent='Index document';}
};

const auditDashboard=document.getElementById('auditDashboard');
const chatArea=document.getElementById('chat');
const composerWrap=document.querySelector('.composer-wrap');
const auditKeyInput=document.getElementById('auditAdminKey');
const uploadKeyInput=document.getElementById('adminKey');
const auditStatus=document.getElementById('auditStatus');
const requestRows=document.getElementById('requestRows');
const ingestionRows=document.getElementById('ingestionRows');
const requestPageSize=25;
let requestOffset=0;
let currentAuditItems=[];
let currentIngestionItems=[];
let auditTotal=0;

function showAuditStatus(message,kind=''){
  auditStatus.textContent=message;
  auditStatus.className=`audit-status ${kind}`.trim();
}
function displayTime(value){
  if(!value)return '—';
  const date=new Date(value);
  return Number.isNaN(date.getTime())?escapeHtml(value):escapeHtml(date.toLocaleString());
}
function numberOrDash(value, suffix=''){
  return value===null||value===undefined||!Number.isFinite(Number(value))?'—':`${Number(value).toLocaleString()}${suffix}`;
}
function auditDateBounds(){
  const from=document.getElementById('auditFrom').value;
  const to=document.getElementById('auditTo').value;
  const result={};
  if(from)result.from_ts=`${from}T00:00:00.000+00:00`;
  if(to)result.to_ts=`${to}T23:59:59.999+00:00`;
  return result;
}
function auditFilters(){
  const filters=auditDateBounds();
  const outcome=document.getElementById('auditOutcome').value;
  const route=document.getElementById('auditRoute').value;
  if(outcome)filters.outcome=outcome;
  if(route)filters.selected_route=route;
  return filters;
}
function queryString(values){
  const query=new URLSearchParams();
  Object.entries(values).forEach(([key,value])=>{if(value!==undefined&&value!==null&&value!=='')query.set(key,value);});
  return query.toString();
}
async function auditGet(path,key){
  const response=await fetch(path,{headers:{'X-Admin-Key':key}});
  const body=await response.json().catch(()=>({}));
  if(!response.ok){
    const message=typeof body.detail==='string'?body.detail:(response.status===401?'Admin key was not accepted.':'Audit data could not be loaded.');
    throw new Error(message);
  }
  return body;
}
function setMetric(id,value){document.getElementById(id).textContent=value;}
function renderMetrics(data){
  const metrics=data.query_requests||{};
  setMetric('metricTotal',numberOrDash(metrics.total));
  setMetric('metricFailureRate',metrics.failure_rate===null||metrics.failure_rate===undefined?'—':`${(Number(metrics.failure_rate)*100).toFixed(1)}%`);
  setMetric('metricPrivateKb',numberOrDash(metrics.private_kb_requests));
  setMetric('metricWeb',numberOrDash(metrics.web_fallback_requests));
  setMetric('metricInsufficient',numberOrDash(metrics.insufficient_evidence));
  setMetric('metricLatency',metrics.average_latency_ms===null||metrics.average_latency_ms===undefined?'—':`${Number(metrics.average_latency_ms).toFixed(0)} ms`);
}
function renderRequests(data){
  currentAuditItems=data.items||[]; auditTotal=data.total||0;
  document.getElementById('requestCount').textContent=`${auditTotal.toLocaleString()} matching request(s)`;
  if(!currentAuditItems.length){
    requestRows.innerHTML='<tr><td colspan="7" class="empty-cell">No requests match these filters.</td></tr>';
  }else{
    requestRows.innerHTML=currentAuditItems.map((item,index)=>{
      const refs=(item.retrieval_results||[]).slice(0,3).map(ref=>`${ref.source||'unknown'}${ref.page?` p.${ref.page}`:''}`).join(', ');
      const status=item.outcome||'unknown';
      return `<tr><td>${displayTime(item.created_at)}</td><td>${escapeHtml(status.replaceAll('_',' '))}</td><td>${escapeHtml(item.selected_route||'—')}</td><td>${escapeHtml(item.source_used||'—')}</td><td>${numberOrDash(item.latency_ms,' ms')}</td><td>${escapeHtml(refs||'—')}</td><td><button class="secondary-btn request-detail-button" data-index="${index}">Details</button></td></tr>`;
    }).join('');
  }
  const page=Math.floor(requestOffset/requestPageSize)+1;
  document.getElementById('requestPage').textContent=`Page ${page}`;
  document.getElementById('previousRequests').disabled=requestOffset<=0;
  document.getElementById('nextRequests').disabled=requestOffset+requestPageSize>=auditTotal;
}
function renderIngestion(data){
  currentIngestionItems=data.items||[];
  document.getElementById('ingestionCount').textContent=`${(data.total||0).toLocaleString()} job(s)`;
  if(!currentIngestionItems.length){
    ingestionRows.innerHTML='<tr><td colspan="6" class="empty-cell">No ingestion jobs match this date range.</td></tr>';
    return;
  }
  ingestionRows.innerHTML=currentIngestionItems.map(item=>{
    const diagnostics=(item.diagnostics||[]).join('; ');
    return `<tr><td>${displayTime(item.created_at)}</td><td>${escapeHtml(item.source_filename||'—')}<br><small>${escapeHtml(item.document_id||'')}</small></td><td>${escapeHtml(item.status||'—')}</td><td>${escapeHtml(item.vision_status||'—')}</td><td>${numberOrDash(item.latency_ms,' ms')}</td><td>${escapeHtml(diagnostics||item.error_category||'—')}</td></tr>`;
  }).join('');
}
function renderRecentErrors(requestErrorItems=[]){
  const requestErrors=requestErrorItems.map(item=>({
    time:item.created_at,title:`Request ${item.request_id||`#${item.id}`} failed`,
    detail:item.error_category||'Request processing failed',
  }));
  const ingestionErrors=currentIngestionItems.filter(item=>['failed','partial'].includes(item.status)).map(item=>({
    time:item.created_at,title:`Ingestion ${item.status}: ${item.source_filename||item.document_id}`,
    detail:[item.error_category,...(item.diagnostics||[])].filter(Boolean).join(' · ')||'Processing warning',
  }));
  const errors=[...requestErrors,...ingestionErrors].sort((a,b)=>new Date(b.time)-new Date(a.time)).slice(0,8);
  const target=document.getElementById('recentErrors');
  target.innerHTML=errors.length?errors.map(item=>`<div class="audit-list-item"><strong>${escapeHtml(item.title)}</strong><br>${displayTime(item.time)}<br>${escapeHtml(item.detail)}</div>`).join(''):'<div class="empty">No recent failures or partial ingestion jobs in this range.</div>';
}
function renderRequestDetail(item){
  if(!item){document.getElementById('requestDetail').innerHTML='<div class="empty">No request selected.</div>';return;}
  const refs=item.retrieval_results||[];
  const stageEntries=Object.entries(item.stage_timings_ms||{}).sort((a,b)=>b[1]-a[1]);
  const timings=stageEntries.map(([name,ms])=>`<li>${escapeHtml(name.replaceAll('_',' '))} <span class="muted">${Number(ms).toFixed(1)} ms</span></li>`).join('');
  const trace=(item.trace||[]).map(step=>`<li>${escapeHtml(step)}</li>`).join('');
  const evidence=item.evidence||{};
  const supporting=new Set(evidence.supporting_chunk_ids||[]);
  const refHtml=refs.length?`<ul class="reference-list">${refs.map(ref=>`<li>${escapeHtml(ref.source||'Unknown source')}${ref.page?` · page ${escapeHtml(String(ref.page))}`:''}${ref.content_type?` · ${escapeHtml(ref.content_type)}`:''}${ref.chunk_id&&supporting.has(ref.chunk_id)?' · accepted by evidence grader':' · retrieval candidate'}</li>`).join('')}</ul>`:'<div class="empty">No private KB chunks retrieved.</div>';
  const webRefs=evidence.web_references||[];
  const webHtml=webRefs.length?`<ul class="reference-list">${webRefs.map(ref=>`<li>${escapeHtml(ref.title)} · ${escapeHtml(ref.url)}</li>`).join('')}</ul>`:'<div class="empty">No web references recorded.</div>';
  document.getElementById('requestDetail').innerHTML=`
    <div class="detail-meta">
      <div><span>Request ID</span>${escapeHtml(item.request_id||`#${item.id}`)}</div>
      <div><span>Conversation</span>${escapeHtml(item.conversation_id||'—')}</div>
      <div><span>Outcome / source</span>${escapeHtml(item.outcome||'—')} · ${escapeHtml(item.source_used||'—')}</div>
      <div><span>Route / latency</span>${escapeHtml(item.selected_route||'—')} · ${numberOrDash(item.latency_ms,' ms')}</div>
      <div><span>Evidence grades</span>KB: ${escapeHtml(evidence.kb_grade||'—')} · Web: ${escapeHtml(evidence.web_grade||'—')}</div>
      <div><span>Rewrites / retries / fallback</span>${numberOrDash(item.query_rewrite_count)} / ${numberOrDash(item.retry_count)} / ${item.web_fallback?'Yes':'No'}</div>
    </div>
    <strong>Workflow timeline</strong><ol class="timeline">${trace||'<li>No trace recorded.</li>'}</ol>
    <strong>Stage timings</strong><ol class="timeline">${timings||'<li>No stage timings recorded.</li>'}</ol>
    <strong>Retrieved source and page references</strong>${refHtml}
    <strong>External references</strong>${webHtml}
    ${item.error_category?`<strong>Error category:</strong> ${escapeHtml(item.error_category)}`:''}`;
}
async function loadAuditDashboard(){
  const key=auditKeyInput.value.trim();
  if(!key){showAuditStatus('Enter the admin key to load live audit data.','error');return;}
  showAuditStatus('Loading audit records…');
  document.getElementById('loadAudit').disabled=true;
  const base=auditFilters();
  const requestParams={...base,limit:requestPageSize,offset:requestOffset};
  const ingestionParams={...auditDateBounds(),limit:25,offset:0};
  try{
    const [metrics,requests,ingestion,recentFailedRequests]=await Promise.all([
      auditGet(`/api/admin/audit/metrics?${queryString(base)}`,key),
      auditGet(`/api/admin/audit?${queryString(requestParams)}`,key),
      auditGet(`/api/admin/audit/ingestion?${queryString(ingestionParams)}`,key),
      auditGet(`/api/admin/audit?${queryString({...auditDateBounds(),selected_route:base.selected_route,outcome:'failed',limit:8,offset:0})}`,key),
    ]);
    renderMetrics(metrics);renderRequests(requests);renderIngestion(ingestion);renderRecentErrors(recentFailedRequests.items||[]);
    showAuditStatus(`Updated from backend at ${new Date().toLocaleTimeString()}.`,'success');
  }catch(error){
    showAuditStatus(error.message||'Audit data could not be loaded.','error');
    ['metricTotal','metricFailureRate','metricPrivateKb','metricWeb','metricInsufficient','metricLatency'].forEach(id=>setMetric(id,'—'));
    document.getElementById('recentErrors').innerHTML='<div class="empty">Audit data is unavailable.</div>';
    requestRows.innerHTML='<tr><td colspan="7" class="empty-cell">Could not load audit records.</td></tr>';
    ingestionRows.innerHTML='<tr><td colspan="6" class="empty-cell">Could not load ingestion jobs.</td></tr>';
    document.getElementById('requestDetail').innerHTML='<div class="empty">Request details are unavailable.</div>';
  }finally{document.getElementById('loadAudit').disabled=false;}
}
function openAuditDashboard(){
  chatArea.classList.add('hidden');composerWrap.classList.add('hidden');auditDashboard.classList.remove('hidden');
  if(uploadKeyInput.value&&!auditKeyInput.value)auditKeyInput.value=uploadKeyInput.value;
}
function closeAuditDashboard(){auditDashboard.classList.add('hidden');chatArea.classList.remove('hidden');composerWrap.classList.remove('hidden');}
document.getElementById('openAudit').addEventListener('click',openAuditDashboard);
document.getElementById('closeAudit').addEventListener('click',closeAuditDashboard);
document.getElementById('loadAudit').addEventListener('click',loadAuditDashboard);
document.getElementById('auditFilters').addEventListener('submit',event=>{event.preventDefault();requestOffset=0;loadAuditDashboard();});
document.getElementById('clearAuditFilters').addEventListener('click',()=>{
  document.getElementById('auditFrom').value='';document.getElementById('auditTo').value='';
  document.getElementById('auditOutcome').value='';document.getElementById('auditRoute').value='';requestOffset=0;loadAuditDashboard();
});
document.getElementById('previousRequests').addEventListener('click',()=>{requestOffset=Math.max(0,requestOffset-requestPageSize);loadAuditDashboard();});
document.getElementById('nextRequests').addEventListener('click',()=>{if(requestOffset+requestPageSize<auditTotal){requestOffset+=requestPageSize;loadAuditDashboard();}});
requestRows.addEventListener('click',event=>{
  const button=event.target.closest('.request-detail-button');
  if(button)renderRequestDetail(currentAuditItems[Number(button.dataset.index)]);
});
auditKeyInput.addEventListener('input',()=>{uploadKeyInput.value=auditKeyInput.value;});
uploadKeyInput.addEventListener('input',()=>{auditKeyInput.value=uploadKeyInput.value;});
