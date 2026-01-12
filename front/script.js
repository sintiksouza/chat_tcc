// ================= CONFIGURAÇÃO DA API =================
const API_BASE = "http://localhost:8000";
const ENDPOINT = "/chatbot/answer";

const K_DEFAULT = 5;
const CULTURA_DEFAULT = "abacaxi";

const API_KEY = "minha_chave_tcc";

// ================= EXEMPLOS POR CATEGORIA =================
const EXAMPLES = {
  duvidas: {
    label: "Dúvidas frequentes",
    items: [
      "Qual é a época ideal da colheita do abacaxi?",
      "Quanto tempo demora para o abacaxi produzir?",
      "Qual espaçamento recomendado para plantio de abacaxi?",
      "Qual é a melhor adubação para o abacaxizeiro?",
      "Como fazer irrigação no abacaxi em períodos secos?"
    ]
  },
  doencas: {
    label: "Doenças",
    items: [
      "Quais são os sintomas de fusariose no abacaxi?",
      "Quais doenças atacam a coroa do abacaxi?",
      "Como prevenir murcha e apodrecimento em mudas?",
      "Como diferenciar doença de deficiência nutricional?"
    ]
  },
  tratamentos: {
    label: "Tratamentos",
    items: [
      "Como controlar fusariose no abacaxi (manejo integrado)?",
      "Quais práticas de manejo reduzem incidência de doenças?",
      "O que fazer quando há podridão do fruto?",
      "Como realizar tratamento de mudas antes do plantio?",
      "Boas práticas de rotação e eliminação de restos culturais?"
    ]
  },
  cultivo: {
    label: "Dicas de cultivo",
    items: [
      "Como preparar o solo para plantio de abacaxi?",
      "Quais condições de clima e solo são ideais para abacaxi?",
      "Como fazer controle de plantas daninhas no abacaxi?",
      "Quando e como fazer indução floral no abacaxi?"
    ]
  },
  campo: {
    label: "Relatos do campo",
    items: [
      "Quais erros mais comuns no cultivo do abacaxi e como evitar?",
      "Como reduzir perdas na colheita e transporte?",
      "Boas práticas para padronizar tamanho e qualidade do fruto?",
      "Como organizar um calendário de manejo do abacaxi?",
      "Quais indicadores de maturação usar na prática?"
    ]
  }
};

// ================= DOM =================
const $ = (id) => document.getElementById(id);

const messages = $("messages");
const sendBtn = $("sendBtn");
const questionInput = $("question");
const chipsBox = $("chips");
const catLabel = $("catLabel");
const examplesHint = $("examplesHint");

// ================= UI =================
function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  msg.appendChild(bubble);
  messages.appendChild(msg);
  messages.scrollTop = messages.scrollHeight;
}

function setActiveCategory(catKey) {
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.cat === catKey);
  });

  catLabel.textContent = `Categoria: ${EXAMPLES[catKey].label}`;

  examplesHint.classList.add("hidden");
  chipsBox.classList.remove("hidden");
  chipsBox.innerHTML = "";

  EXAMPLES[catKey].items.forEach(q => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = q;

    chip.addEventListener("click", () => {
      questionInput.value = q;
      questionInput.focus();
    });

    chipsBox.appendChild(chip);
  });
}

// ================= API =================
async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  questionInput.value = "";
  sendBtn.disabled = true;

  if (!API_KEY) {
    addMessage(
      "Erro: API_KEY não definida no front. Abra o arquivo script.js",
      "bot"
    );
    sendBtn.disabled = false;
    return;
  }

  try {
    const res = await fetch(API_BASE + ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
      },
      body: JSON.stringify({
        query: question,
        k: K_DEFAULT,
        cultura: CULTURA_DEFAULT
      })
    });

    if (!res.ok) {
      const text = await res.text();
      addMessage(`Erro HTTP ${res.status}\n${text}`, "bot");
      return;
    }

    const data = await res.json();

    // Apenas a resposta (sem fontes)
    addMessage(data.answer || "Sem resposta.", "bot");

  } catch (err) {
    addMessage("Falha de conexão com o servidor.", "bot");
  } finally {
    sendBtn.disabled = false;
  }
}

// ================= EVENTOS =================
sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuestion();
});

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => setActiveCategory(btn.dataset.cat));
});
