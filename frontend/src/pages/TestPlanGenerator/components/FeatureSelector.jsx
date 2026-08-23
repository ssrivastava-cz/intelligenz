import Select from "../../../components/Select/Select.jsx";
import { useSourceOfTruthFeatures } from "../../../hooks/useSourceOfTruthFeatures.js";

/** Feature dropdown, backed by the real Source of Truth feature list. */
function FeatureSelector({ value, onChange }) {
  const { features, isLoading, error } = useSourceOfTruthFeatures();

  return (
    <Select
      id="feature-selector"
      label="Feature"
      required
      value={value}
      onChange={onChange}
      options={features}
      placeholder={isLoading ? "Loading features…" : error ? "Unable to load features" : "Select a feature…"}
    />
  );
}

export default FeatureSelector;
