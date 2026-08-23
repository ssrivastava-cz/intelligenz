import TextInput from "../../../components/TextInput/TextInput.jsx";

/**
 * Redmine ticket ID input — accepts one or more comma-separated IDs.
 * Optional: generation can run from the selected Feature alone, so
 * this field is never required.
 */
function TicketIdField({ value, onChange }) {
  return (
    <TextInput
      id="ticket-id"
      label="Redmine Ticket ID"
      value={value}
      onChange={onChange}
      placeholder="e.g. 12345 or 12345,12346"
      hint="Optional. Accepts comma-separated ticket IDs."
    />
  );
}

export default TicketIdField;
