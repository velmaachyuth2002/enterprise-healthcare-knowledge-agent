from pydantic import BaseModel

# Placeholder content only — will be replaced once real policy documents
# exist and are served through retrieval instead of a hardcoded lookup.
_POLICIES: dict[str, tuple[str, str]] = {
    "provider_onboarding": (
        "Provider Onboarding Policy",
        "Placeholder: new provider organizations must complete identity "
        "verification and sign a data processing agreement before onboarding "
        "begins.",
    ),
}


class PolicyLookupInput(BaseModel):
    topic: str


class PolicyLookupResult(BaseModel):
    found: bool
    policy_name: str | None = None
    content: str | None = None


class PolicyTool:
    name = "policy_lookup"
    description = "Look up an internal company policy by topic."

    def run(self, params: PolicyLookupInput) -> PolicyLookupResult:
        entry = _POLICIES.get(self._normalize(params.topic))
        if entry is None:
            return PolicyLookupResult(found=False)

        policy_name, content = entry
        return PolicyLookupResult(found=True, policy_name=policy_name, content=content)

    def known_topics(self) -> list[str]:
        """Valid topic keys this tool can look up.

        Callers extract a topic from a question by matching against this
        list, rather than keeping their own separate keyword table that
        could drift out of sync with what this tool actually has.
        """
        return list(_POLICIES.keys())

    @staticmethod
    def _normalize(topic: str) -> str:
        return topic.strip().lower().replace(" ", "_")
