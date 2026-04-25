function LabeledSelectField({
  id,
  label,
  value,
  onChange,
  options,
  emptyLabel
}) {
  const hasOptions = Array.isArray(options) && options.length > 0;

  return (
    <>
      <label className="construction-field-label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        className="construction-select"
        value={value}
        onChange={onChange}
        disabled={!hasOptions}
      >
        {!hasOptions && <option value="">{emptyLabel}</option>}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </>
  );
}

export default LabeledSelectField;
