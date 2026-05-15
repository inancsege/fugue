package fugue

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// reflectInputSchema produces a JSON Schema Draft 7-ish object from a Go
// type. The top-level type must be a struct (or pointer to struct). The
// returned RawMessage encodes a JSON object with "type": "object",
// "properties", and "required" keys.
//
// Reflection is best-effort: see the package docs for the supported type
// list. Unsupported types return an error whose message includes the field
// path that failed; callers (typically the Tool[In, Out] constructor)
// should panic on this error since it represents a programming bug.
func reflectInputSchema(t reflect.Type) (json.RawMessage, error) {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("fugue.Tool: input type must be a struct (or pointer to struct), got %s", t.Kind())
	}
	return structSchema(t, "")
}

// structSchema builds a JSON Schema object for a struct type. path is the
// dotted field path used for error messages (empty at the top level).
//
// Properties are emitted in struct-declaration order via a manual JSON
// build — encoding/json sorts map keys alphabetically, and we want stable,
// declaration-ordered output for readability.
func structSchema(t reflect.Type, path string) (json.RawMessage, error) {
	type prop struct {
		name   string
		schema json.RawMessage
	}
	var props []prop
	var required []string

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name, omitempty, skip := parseJSONTag(f)
		if skip {
			continue
		}
		fieldPath := path + "." + f.Name
		if path == "" {
			fieldPath = f.Name
		}
		propSchema, fieldRequired, err := fieldSchema(f, fieldPath)
		if err != nil {
			return nil, err
		}
		props = append(props, prop{name: name, schema: propSchema})
		if !omitempty && fieldRequired {
			required = append(required, name)
		}
	}

	var b strings.Builder
	b.WriteString(`{"type":"object"`)
	if len(props) > 0 {
		b.WriteString(`,"properties":{`)
		for i, p := range props {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(jsonQuote(p.name))
			b.WriteByte(':')
			b.Write(p.schema)
		}
		b.WriteByte('}')
	}
	if len(required) > 0 {
		b.WriteString(`,"required":[`)
		for i, name := range required {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(jsonQuote(name))
		}
		b.WriteByte(']')
	}
	b.WriteByte('}')
	return json.RawMessage(b.String()), nil
}

// fieldSchema builds a JSON Schema for a single struct field, and reports
// whether the field is required (before omitempty adjustments handled in
// structSchema). In this task only primitives are supported; later tasks
// extend primitiveSchema to handle slices, maps, pointers, structs, etc.
func fieldSchema(f reflect.StructField, path string) (schema json.RawMessage, required bool, err error) {
	desc := f.Tag.Get("fugue")
	primitive, err := primitiveSchema(f.Type, path)
	if err != nil {
		return nil, false, err
	}
	if desc != "" {
		primitive = injectDescription(primitive, desc)
	}
	return primitive, true, nil
}

// primitiveSchema handles the scalar types only. Other kinds return an error
// in this task and will be supported in later tasks.
func primitiveSchema(t reflect.Type, path string) (json.RawMessage, error) {
	switch t.Kind() {
	case reflect.String:
		return json.RawMessage(`{"type":"string"}`), nil
	case reflect.Bool:
		return json.RawMessage(`{"type":"boolean"}`), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return json.RawMessage(`{"type":"integer"}`), nil
	case reflect.Float32, reflect.Float64:
		return json.RawMessage(`{"type":"number"}`), nil
	}
	return nil, fmt.Errorf("fugue.Tool: field %q has unsupported type %s", path, t.String())
}

// parseJSONTag returns (name, omitempty, skip):
//   - name = explicit `json:"name"` or the struct field name when the tag is
//     absent.
//   - omitempty = true if the json tag includes ",omitempty".
//   - skip = true if the json tag is "-".
func parseJSONTag(f reflect.StructField) (name string, omitempty bool, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	if tag == "" {
		return f.Name, false, false
	}
	parts := strings.Split(tag, ",")
	name = parts[0]
	if name == "" {
		name = f.Name
	}
	for _, p := range parts[1:] {
		if p == "omitempty" {
			omitempty = true
		}
	}
	return name, omitempty, false
}

// injectDescription returns schema with a "description" key inserted at the
// start. Caller guarantees schema is a JSON object; this function inserts
// after the opening "{" so the description appears first.
func injectDescription(schema json.RawMessage, desc string) json.RawMessage {
	s := string(schema)
	if !strings.HasPrefix(s, "{") {
		return schema
	}
	inner := s[1:]
	descPart := `"description":` + jsonQuote(desc)
	if len(inner) > 1 {
		descPart += ","
	}
	return json.RawMessage("{" + descPart + inner)
}

// jsonQuote returns a JSON-quoted string. Used in place of strconv.Quote
// since this package builds schema JSON manually for deterministic key order.
func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
