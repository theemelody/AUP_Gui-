interface SelectOption {
  value: string;
  label: string;
}

interface LabeledSelectFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: SelectOption[];
  emptyLabel?: string;
}

function LabeledSelectField({
  id,
  label,
  value,
  onChange,
  options,
  emptyLabel,
}: LabeledSelectFieldProps) {
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
