from app.tools.policy_tool import PolicyLookupInput, PolicyTool


def test_returns_policy_for_known_topic() -> None:
    tool = PolicyTool()

    result = tool.run(PolicyLookupInput(topic="provider_onboarding"))

    assert result.found is True
    assert result.policy_name == "Provider Onboarding Policy"
    assert result.content is not None


def test_normalizes_topic_casing_and_spacing() -> None:
    tool = PolicyTool()

    result = tool.run(PolicyLookupInput(topic="Provider Onboarding"))

    assert result.found is True
    assert result.policy_name == "Provider Onboarding Policy"


def test_returns_not_found_for_unknown_topic_without_raising() -> None:
    tool = PolicyTool()

    result = tool.run(PolicyLookupInput(topic="vacation_policy"))

    assert result.found is False
    assert result.content is None
