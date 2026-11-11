import os

from mistralai import agents

DEFAULT_LIBRARY_IDS = [
    "01a1a290-bb2a-75dc-8c20-55a4cd86d129",
    "01a19db3-45cb-73e9-94f4-7f7df10d73fd",
    "01a19da6-a4fc-72a9-860b-9aacbeb1f2fe",
    "01a197ec-ecec-7c7450-a7f3-409c19965c1",
    "01a17965-d194-73a1-8cc3-dd24288d8e8c",
    "01a1795d-6054-704c-8a65-d4fe041868bc",
    "01a171f9-5fa1-7048-99e2-cba5fffde894",
    "01a16e39-2f3c-72c6-889a-2236b5e17dd5",
    "01a168eb-2c62-7401-ae88-872fbcf2adcd",
    "01a167d9-be34-75f8-8d4e-94904eca4e87",
    "01a16509-d6f5-7051-9dad-9c905d5e303e",
    "01a163e9-1027-7512-b105-49a3cbe63f22",
]

INSTRUCTIONS = """Tu es un trader professionnel spécialisé en scalping, et l'assistant personnel de l'utilisateur.

- Réponds toujours en français, de manière concise et directe.
- Ton objectif permanent est d'aider l'utilisateur à préparer, analyser et comprendre ses trades de scalping (setups, gestion du risque, timing, psychologie de marché), en t'appuyant sur ses bibliothèques Vibe Work.
- Avant de répondre à une question qui peut être couverte par les documents de l'utilisateur, appelle l'outil search_libraries pour fonder ta réponse sur ses propres documents, et mentionne les bibliothèques ou documents utilisés lorsque c'est pertinent.
- Tu vois toute la conversation en cours : tu te souviens de l'objectif de trading de l'utilisateur, de ses préférences, de ses décisions et de tout ce qui a été dit dans cette session. Les questions de suivi se comprennent dans ce contexte, sans que l'utilisateur ait à se réexpliquer. Ne lui demande jamais de résumer ce que vous avez déjà établi.
- Tu n'exécutes jamais de code : tu n'as aucun environnement sandbox, tu réponds uniquement en langage naturel. Aucun code arbitraire n'est exécuté, jamais.
- Tu ne passes aucun ordre et tu ne gères aucun compte : tu es un outil d'analyse et de discussion. Lorsque la conversation porte sur une décision de trading réelle, conclus par une courte mise en garde : ce n'est pas un conseil financier personnalisé.
"""


def _library_ids() -> list[str]:
    raw = os.environ.get("SCALPING_LIBRARY_IDS", "")
    if raw.strip():
        return [part.strip() for part in raw.split(",") if part.strip()]
    return list(DEFAULT_LIBRARY_IDS)


def _client():
    if os.environ.get("MISTRAL_SA_TOKEN_PATH"):
        from mistralai.workflows.client import get_mistral_client

        return get_mistral_client(use_executor_credentials=True)
    from mistralai.client import Mistral

    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


@agents.tool
async def search_libraries(query: str) -> str:
    """Recherche dans les bibliothèques Vibe Work de l'utilisateur (stratégies, backtests, informations sur les symboles, notes de trading) les passages pertinents pour la question. Appelle cet outil avant toute réponse susceptible d'être couverte par ces documents."""
    response = await _client().beta.conversations.start_async(
        model="mistral-large-4",
        inputs=query,
        tools=[{"type": "document_library", "library_ids": _library_ids()}],
        store=False,
    )
    parts: list[str] = []
    for entry in response.outputs:
        content = getattr(entry, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
            continue
        for chunk in content:
            text = getattr(chunk, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts) if parts else "Aucun passage pertinent trouvé dans les bibliothèques."


agent = agents.Agent(
    model="mistral-large-4",
    name="scalping-trader",
    description="Trader professionnel en scalping, connecté aux bibliothèques Vibe Work de l'utilisateur. Répond en français, en se souvenant de toute la conversation.",
    instructions=INSTRUCTIONS,
    tools=[search_libraries],
    temperature=0.2,
    reasoning_effort="high",
)
