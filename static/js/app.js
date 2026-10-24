const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const conversationStorageKey = 'agentic-rag-conversation-id';
let conversationId = null;
let conversationReady = Promise.resolve();

function escapeHtml(s=''){return s.replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
function addMessage(role, text, source='', citations=[]){
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  wrap.innerHTML=`<div class="avatar">${role==='assistant'?'AI':'You'}</div><div class="bubble">${formatText(text)}${source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:''}${citeHtml}</div>`;
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}
function showWelcome(){chat.innerHTML='';addMessage('assistant','How can I help with IT support? Try an IT support question or ask about current vendor information.');}
async function createConversation(){
  const res=await fetch('/api/conversations',{method:'POST'}); const data=await res.json();
  if(!res.ok) throw new Error(data.detail||'Could not start a conversation');
  conversationId=data.conversation_id; localStorage.setItem(conversationStorageKey,conversationId);
  showWelcome(); renderTrace([]); sourceUsed.textContent='—';
}
async function resumeConversation(){
  const saved=localStorage.getItem(conversationStorageKey);
  if(!saved){await createConversation();return;}
  const res=await fetch(`/api/conversations/${encodeURIComponent(saved)}`);
  if(res.status===404){localStorage.removeItem(conversationStorageKey);await createConversation();return;}
  if(!res.ok) throw new Error('Could not resume the conversation');
  const data=await res.json(); conversationId=data.conversation_id; chat.innerHTML='';
  if(!data.messages.length) showWelcome(); else data.messages.forEach(m=>addMessage(m.role,m.content));
}
async function askAgent(q){
  await conversationReady;
  if(!conversationId) await createConversation();
  addMessage('user',q); question.value=''; renderTrace(['Running LangGraph workflow...']); sourceUsed.textContent='Running';
  const btn=form.querySelector('button'); btn.disabled=true;
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,conversation_id:conversationId})});
    const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Request failed');
    conversationId=data.conversation_id; localStorage.setItem(conversationStorageKey,conversationId);
    addMessage('assistant',data.answer,data.source_used,data.citations||[]); renderTrace(data.trace||[]); sourceUsed.textContent=data.source_used;
  }catch(e){addMessage('assistant',`Error: ${e.message}`); renderTrace(['Request failed']); sourceUsed.textContent='Error';}
  finally{btn.disabled=false;}
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
document.querySelectorAll('.example').forEach(b=>b.addEventListener('click',()=>askAgent(b.textContent.trim())));
document.getElementById('newConversation').addEventListener('click',async()=>{
  await conversationReady;
  try{await createConversation();}catch(e){addMessage('assistant',`Error: ${e.message}`);}
});
conversationReady=resumeConversation().catch(e=>addMessage('assistant',`Error: ${e.message}`));

const modal=document.getElementById('uploadModal');
document.getElementById('openUpload').onclick=()=>modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick=()=>modal.classList.add('hidden');
document.getElementById('uploadBtn').onclick=async()=>{
  const file=document.getElementById('fileInput').files[0]; const key=document.getElementById('adminKey').value; const status=document.getElementById('uploadStatus');
  if(!file){status.textContent='Choose a file first.'; return;}
  status.textContent='Indexing document...'; const fd=new FormData(); fd.append('file',file);
  try{
    const r=await fetch('/api/ingest',{method:'POST',headers:{'X-Admin-Key':key},body:fd});
    const d=await r.json(); if(!r.ok) throw new Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||'Upload failed'));
    const warningText=d.warnings?.length?`\nWarnings: ${d.warnings.join(' ')}`:'';
    status.textContent=`Indexed ${d.file}: ${d.chunks} chunks from ${d.pages} page(s).${warningText}`;
  }catch(e){status.textContent=`Error: ${e.message}`;}
};
