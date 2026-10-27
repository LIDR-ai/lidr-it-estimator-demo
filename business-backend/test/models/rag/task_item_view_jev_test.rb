require "test_helper"

# Graph flow: who accepted a task's retrieved hours in the Jev triage — the real Jev or
# the labelled heuristic stand-in used while Jev is unavailable (JEV_FALLBACK=simulated).
class RagTaskItemViewJevTest < ActiveSupport::TestCase
  test "real Jev acceptance" do
    task = Rag::TaskItemView.from_hash("name" => "Login", "accepted_by" => "jev")
    assert task.jev_accepted?
    assert_not task.jev_simulated?
  end

  test "simulated acceptance is still an acceptance, but labelled" do
    task = Rag::TaskItemView.from_hash("name" => "Login", "accepted_by" => "jev_simulated")
    assert task.jev_accepted?
    assert task.jev_simulated?
  end

  test "no triage" do
    task = Rag::TaskItemView.from_hash("name" => "Login")
    assert_not task.jev_accepted?
    assert_not task.jev_simulated?
  end
end
