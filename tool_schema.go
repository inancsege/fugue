package fugue

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"
)

// timeTimeType is cached for cheap equality checks in typeToSchema.
var timeTimeType = reflect.TypeOf(time.Time{})

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
	visited := map[reflect.Type]bool{}
	return structSchemaVisited(t, "", visited)
}

// structSchemaVisited builds a JSON Schema object for a struct type. path is
// the dotted field path used for error messages (empty at the top level).
// visited tracks struct types currently on the recursion stack to detect
// recursive types; the defer-delete ensures sibling fields of the same type
// are not falsely flagged.
//
// Properties are emitted in struct-declaration order via a manual JSON
// build — encoding/json sorts map keys alphabetically, and we want stable,
// declaration-ordered output for readability.
func structSchemaVisited(t reflect.Type, path string, visited map[reflect.Type]bool) (json.RawMessage, error) {
	if visited[t] {
		return nil, fmt.Errorf("fugue.Tool: field %q is a recursive type %s; recursive types are not supported in v1 — use RawTool", path, t.String())
	}
	visited[t] = true
	defer delete(visited, t)

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
		propSchema, fieldRequired, err := fieldSchemaVisited(f, fieldPath, visited)
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

// fieldSchemaVisited builds a JSON Schema for a single struct field, and
// reports whether the field is required before omitempty adjustment. Pointer
// fields are non-required (pointer semantics: nil means absent). Description
// from the `fugue:` tag and enum from the `fugueEnum:` tag are layered into
// the resulting schema.
func fieldSchemaVisited(f reflect.StructField, path string, visited map[reflect.Type]bool) (schema json.RawMessage, required bool, err error) {
	desc := f.Tag.Get("fugue")
	enumTag := f.Tag.Get("fugueEnum")
	fieldType := f.Type
	required = fieldType.Kind() != reflect.Pointer

	// fugueEnum is only valid on string fields (also on *string via pointer).
	checkType := fieldType
	if checkType.Kind() == reflect.Pointer {
		checkType = checkType.Elem()
	}
	if enumTag != "" && checkType.Kind() != reflect.String {
		return nil, false, fmt.Errorf("fugue.Tool: field %q: fugueEnum is only valid on string fields, got %s", path, checkType.Kind())
	}

	s, err := typeToSchema(fieldType, path, visited)
	if err != nil {
		return nil, false, err
	}
	if enumTag != "" {
		s = injectEnum(s, enumTag)
	}
	if desc != "" {
		s = injectDescription(s, desc)
	}
	return s, required, nil
}

// typeToSchema produces a JSON Schema fragment for a Go type. Container
// kinds (slice, map, pointer) recurse. Returns an error whose message
// includes the field path for unsupported types.
//
// json.RawMessage is special-cased to the "any JSON" empty-object schema
// so users can pass arbitrary JSON through. time.Time is explicitly rejected
// with a helpful message pointing to string/int alternatives.
func typeToSchema(t reflect.Type, path string, visited map[reflect.Type]bool) (json.RawMessage, error) {
	// json.RawMessage is []byte under the hood — match by exact type so
	// it doesn't fall into the generic slice path.
	if t == reflect.TypeOf(json.RawMessage(nil)) {
		return json.RawMessage(`{}`), nil
	}
	// time.Time is a struct but its zero-value semantics aren't expressible
	// in JSON Schema, so reject explicitly and point users at RawTool.
	if t == timeTimeType {
		return nil, fmt.Errorf("fugue.Tool: field %q is time.Time; time.Time is not supported in v1 — use string (RFC3339) or int64 (unix), or use RawTool", path)
	}
	switch t.Kind() {
	case reflect.String:
		return json.RawMessage(`{"type":"string"}`), nil
	case reflect.Bool:
		return json.RawMessage(`{"type":"boolean"}`), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return json.RawMessage(`{"type":"integer"}`), nil
	case reflect.Float32, reflect.Float64:
		return json.RawMessage(`{"type":"number"}`), nil
	case reflect.Pointer:
		return typeToSchema(t.Elem(), path, visited)
	case reflect.Slice, reflect.Array:
		items, err := typeToSchema(t.Elem(), path+"[]", visited)
		if err != nil {
			return nil, err
		}
		return json.RawMessage(`{"type":"array","items":` + string(items) + `}`), nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("fugue.Tool: field %q has map with non-string key %s; JSON object keys must be strings", path, t.Key().String())
		}
		val, err := typeToSchema(t.Elem(), path+"[v]", visited)
		if err != nil {
			return nil, err
		}
		return json.RawMessage(`{"type":"object","additionalProperties":` + string(val) + `}`), nil
	case reflect.Struct:
		return structSchemaVisited(t, path, visited)
	case reflect.Interface:
		return nil, fmt.Errorf("fugue.Tool: field %q has interface type; interface fields are ambiguous — use json.RawMessage for arbitrary JSON or a concrete struct", path)
	case reflect.Chan:
		return nil, fmt.Errorf("fugue.Tool: field %q has chan type; channels cannot be represented in JSON Schema — use RawTool", path)
	case reflect.Func:
		return nil, fmt.Errorf("fugue.Tool: field %q has func type; functions cannot be represented in JSON Schema — use RawTool", path)
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

// injectEnum inserts an "enum" key into a string schema. The enum is parsed
// from a comma-separated list (the fugueEnum tag value); each value is
// trimmed and emitted as a JSON string. Order is preserved from the tag.
func injectEnum(schema json.RawMessage, tag string) json.RawMessage {
	values := strings.Split(tag, ",")
	var enumJSON strings.Builder
	enumJSON.WriteString(`"enum":[`)
	for i, v := range values {
		if i > 0 {
			enumJSON.WriteByte(',')
		}
		enumJSON.WriteString(jsonQuote(strings.TrimSpace(v)))
	}
	enumJSON.WriteByte(']')

	s := string(schema)
	if !strings.HasPrefix(s, "{") {
		return schema
	}
	inner := s[1:]
	insert := enumJSON.String()
	if len(inner) > 1 {
		insert += ","
	}
	return json.RawMessage("{" + insert + inner)
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
