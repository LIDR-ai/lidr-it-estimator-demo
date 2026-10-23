# Mirror of the FastAPI ``TaskItem`` plus the Session 10 per-task hours fields.
# Once estimated, a task carries the hours derived by vector search
# (``estimated_hours`` + ``hours_reliability`` + ``has_match``) and the human-set
# ``rate_eur_per_hour``. ``engineer_days`` stays for the legacy Session 9
# single-shot path (it is nil in the structure-only flow). Per-task citations
# (``sources``) were dropped: the structure is a free decomposition with no
# retrieval, so those ids were always empty and never fed the hours computation.
module Rag
  class TaskItemView
    include ActiveModel::Model
    include ActiveModel::Attributes

    attribute :name, :string
    attribute :description, :string
    attribute :engineer_days, :integer
    # Session 10 hours flow. FLOAT, not integer: the graph emits halves (a gate-2 input is
    # step="0.5" and the service rounds to one decimal). Declaring this as :integer silently
    # truncated 7.5 h to 7, so the per-module subtotals under-reported the real hours — and
    # any money derived from them would inherit the same loss.
    attribute :estimated_hours, :float
    attribute :hours_reliability, :float
    attribute :rate_eur_per_hour, :integer
    attribute :has_match, :boolean, default: true
    # Graph flow: "jev" when Jev's triage accepted the retrieved hours as is (they never
    # went to the recovery agent). Nil everywhere else.
    attribute :accepted_by, :string

    attr_reader :hours_range

    def self.from_hash(hash)
      new(hash || {})
    end

    def initialize(attributes = {})
      stringified = (attributes || {}).transform_keys(&:to_s)
      # Session 11: a contradictory-sources hours range {low, high, reason}.
      @hours_range = Rag::HourRangeView.from_hash(stringified.delete("hours_range"))
      super(stringified.slice(
        "name", "description", "engineer_days",
        "estimated_hours", "rate_eur_per_hour", "hours_reliability", "has_match",
        "accepted_by"
      ))
    end

    # Session 11: the historical sources disagreed, so the hours are a range.
    def contradicted? = hours_range&.present?

    # Red flag: no historical analog was found, so no hours were derived.
    def flagged? = has_match == false

    def jev_accepted? = accepted_by == "jev"

    def reliability_pct = hours_reliability ? (hours_reliability * 100).round : nil

    # Traffic-light band for the UI: red = no match, amber = weak, green = strong.
    def reliability_band
      return :red if flagged?
      return :unknown if hours_reliability.nil?

      hours_reliability >= 0.66 ? :green : :amber
    end

    def cost_eur
      return 0 if estimated_hours.nil? || rate_eur_per_hour.nil?

      estimated_hours * rate_eur_per_hour
    end
  end
end
